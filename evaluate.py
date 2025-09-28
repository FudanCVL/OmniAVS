import os
import os.path as osp
import re
import json
import random
import argparse
import warnings
warnings.filterwarnings("ignore")
from functools import partial
from collections import defaultdict

import numpy as np
import torch
from torch import distributed as dist
from PIL import Image
from tqdm import tqdm
import pycocotools.mask as maskUtils

from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform
import cv2
from skimage.morphology import disk
import math

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param:
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    NF, bsz, H, W = pred.shape
    pred = pred.view(NF*bsz, H, W).cuda()
    target = target.view(NF*bsz, H, W).cuda()
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)

    pred = pred.int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    iou = torch.sum((inter + eps) / (union + eps)) / N

    return iou


def _eval_pr(y_pred, y, num, device='cuda'):
    y_pred = y_pred.to(device)
    y = y.to(device)
    
    prec = torch.zeros(num, device=device)
    recall = torch.zeros(num, device=device)
    thlist = torch.linspace(0, 1 - 1e-10, num, device=device)

    y_temp = (y_pred.unsqueeze(0) >= thlist.view(-1, 1, 1, 1)).float()
    
    y_sum = y.sum() + 1e-20
    
    tp = (y_temp * y).sum(dim=(1,2,3))
    y_temp_sum = y_temp.sum(dim=(1,2,3)) + 1e-20
    
    prec = tp / y_temp_sum
    recall = tp / y_sum

    return prec, recall


def Eval_Fmeasure(pred, gt, pr_num=255):
    rank, _ = get_dist_info()
    device = f"cuda:{rank}"
    
    N = pred.size(0)
    beta2 = 0.3
    avg_f, img_num = torch.zeros(pr_num).to(device), 0

    for img_id in range(N):
        if torch.mean(gt[img_id]) == 0.0:
            continue
            
        prec, recall = _eval_pr(pred[img_id], gt[img_id], pr_num, device=device)
        
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score = torch.nan_to_num(f_score, nan=0.0)
        
        avg_f += f_score.to(device)
        img_num += 1

    score = avg_f / img_num if img_num > 0 else torch.zeros(pr_num).to(device)
    return score.max().item()


def db_eval_boundary(annotation, segmentation, bound_th=0.002):
    assert annotation.shape == segmentation.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :])
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def f_measure(foreground_mask, gt_mask, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    foreground_mask[foreground_mask > 0] = 1
    gt_mask[gt_mask > 0] = 1
    fg_boundary = _seg2bmap(foreground_mask)
    gt_boundary = _seg2bmap(gt_mask)

    fg_dil = cv2.dilate(fg_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F

def _seg2bmap(seg, width=None, height=None):

    # seg[seg > 0] = 1
    # seg = seg.astype

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap



def BLEU(gt, pred):
    """
    Calculate the BLEU score between the ground truth (gt) and the prediction (pred).
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # nltk.download('wordnet')

    gt_tokens = [gt.split()]
    pred_tokens = pred.split()

    smoothie = SmoothingFunction().method4
    weights = (1, 0, 0, 0)  # Only BLEU-1
    bleu_score = sentence_bleu(gt_tokens, pred_tokens, smoothing_function=smoothie)

    return bleu_score


def METEOR(gt, pred):
    """
    Calculate the METEOR score between the ground truth (gt) and the prediction (pred).
    """
    from nltk.translate.meteor_score import single_meteor_score

    gt_tokens = gt.split()
    pred_tokens = pred.split()
    meteor = single_meteor_score(gt_tokens, pred_tokens)

    return meteor


def sample_frames(N, K, strategy='uniform'):
    if K < 2 or K > N:
        raise ValueError("K must be at least 2 and less than or equal to N.")

    selected_frames = [0, N-1]
    middle_frames_needed = K - 2

    if middle_frames_needed > 0:
        middle_frames = list(range(1, N-1))
        if strategy == 'uniform':
            interval = len(middle_frames) // middle_frames_needed
            selected_frames.extend(middle_frames[i * interval] for i in range(middle_frames_needed))

        elif strategy == 'random':
            if middle_frames_needed > len(middle_frames):
                raise ValueError("Not enough middle frames to sample.")
            selected_frames.extend(random.sample(middle_frames, middle_frames_needed))
        else:
            raise ValueError("Invalid sampling strategy. Choose from 'uniform', 'random'")

    return sorted(selected_frames)


ds_collections = {
    'text': ("omniavs", "text"),
    'speech': ("omniavs", "speech"),
    "text_sound": ("omniavs", "text_sound"),
    "speech_sound": ("omniavs", "speech_sound"),
    "text_image": ("omniavs", "text_image"),
    "speech_image": ("omniavs", "speech_image"),
    "text_sound_image": ("omniavs", "text_sound_image"), 
    "speech_sound_image": ("omniavs", "speech_sound_image"),
}

ROOT_DIR = "playground/data/segmentation/omniavs"

PROMPT_DICT = {
    'text': {
        'base': "Please segment the object this sentence describes in this video: <ref>{expression}</ref>",
        'with_expl': "Please segment the object this sentence describes in this video: <ref>{expression}</ref>, and explain why."
    },
    'text_image': {
        'base': "Please segment the object this sentence describes in this video: <ref>{expression}</ref>",
        'with_expl': "Please segment the object this sentence describes in this video: <ref>{expression}</ref>, and explain why."
    },
    'text_sound': {
        'base': "Please segment the object this sentence describes in this video: <ref>{expression}</ref>",
        'with_expl': "Please segment the object this sentence describes in this video: <ref>{expression}</ref>, and explain why."
    },
    'text_sound_image': {
        'base': "Please segment the object this sentence describes in this video: <ref>{expression}</ref>",
        'with_expl': "Please segment the object this sentence describes in this video: <ref>{expression}</ref>, and explain why."
    },
    'speech': {
        'base': "Please segment the object this sentence describes in this video: <ref><audio></ref>",
        'with_expl': "Please segment the object this sentence describes in this video: <ref><audio></ref>, and explain why."
    },
    'speech_sound': {
        'base': "Please segment the object this sentence describes in this video: <ref><audio></ref>, referring audio is <audio>.",
        'with_expl': "Please segment the object this sentence describes in this video: <ref><audio></ref>, referring audio is <audio>, and explain why."
    },
    'speech_image': {
        'base': "Please segment the object this sentence describes in this video: <ref><audio></ref>, referring image is <image>",
        'with_expl': "Please segment the object this sentence describes in this video: <ref><audio></ref>, referring image is <image>, and explain why."
    },
    'speech_sound_image': {
        'base': "Please segment the object this sentence describes in this video: <ref><audio></ref>, referring audio is <audio>, referring image is <image>",
        'with_expl': "Please segment the object this sentence describes in this video: <ref><audio></ref>, referring audio is <audio>, referring image is <image>, and explain why."
    }
}


def collate_fn(batches, tokenizer=None):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    texts = [_['text'] for _ in batches]
    masks = [_['masks'] for _ in batches]
    hws = [_['hw'] for _ in batches]
    splits = [_['split'] for _ in batches]
    frame_flag_lists = [_['frame_flag_list'] for _ in batches]
    levels = [_['level'] for _ in batches]
    audio_inputs = [_['audio_input'] for _ in batches]
    expls = [_['expl'] for _ in batches]
    exp_ids = [_['exp_id'] for _ in batches]
    video_names = [_['video_name'] for _ in batches]
    return pixel_values, texts, masks, hws, splits, frame_flag_lists, levels, audio_inputs, expls, exp_ids, video_names


class OmniAVSDataset(torch.utils.data.Dataset):
    def __init__(self, ds_name, split, prompt=None, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.ds_name = ds_name
        self.split = split
        
        self.raw_data = []
        with open(os.path.join(ROOT_DIR, "meta_expressions_test_deduplicated.jsonl"), 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if data['expr_type'] == self.split:
                    self.raw_data.append(data)

        # self.raw_data = self.raw_data[:30]
        
        
        self.video_dict = json.load(open(os.path.join(ROOT_DIR, "video_dict_new.json"), 'r'))
        
        self.referring_image_path = osp.join(ROOT_DIR, 'referrings', 'referring_images')
        self.referring_speech_path = osp.join(ROOT_DIR, 'referrings', 'referring_speeches')
        self.referring_sound_path = osp.join(ROOT_DIR, 'referrings', 'referring_sounds')
        
        self.sample_frames = 32
        self.dense_sample_frames = 4
        
        self.image_path = osp.join(ROOT_DIR, 'images')
        self.audio_path = osp.join(ROOT_DIR, 'audios')
        
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        anno_info = self.raw_data[idx]
        exp_id = anno_info['exp_id']
        video_name = anno_info['video_name']
        expr = anno_info['expr']
        obj_id = anno_info['obj_id']
        level = ""
        expr_type = anno_info['expr_type']
        expl = anno_info['expl']
        referring_image = anno_info['referring_image']
        referring_sound = anno_info['referring_sound']
        referring_speech = anno_info['referring_speech']
        
        frames_dir = osp.join(self.image_path, video_name)
        audio_path = osp.join(self.audio_path, video_name + '.mp3')
        
        video_info = self.video_dict[video_name]
        frame_num = len(video_info['file_name'])
        file_name = video_info['file_name']
        
        image_list = [osp.join(frames_dir, file_name[i]) for i in range(frame_num)]
        first_frame = Image.open(image_list[0])
        height, width = first_frame.height, first_frame.width
        
        if len(obj_id) == 0:
            null_flag = True
            mask_list = [np.zeros((height, width), dtype=np.uint8)] * frame_num
        else:
            object_mask_list = []
            for i in range(len(obj_id)):
                try:
                    mask_dict = video_info["mask_dict"].get(int(obj_id[i]), video_info["mask_dict"][str(obj_id[0])])
                except:
                    continue
                
                object_mask_list.append(mask_dict)
            
            mask_list = []
            for frame_idx in range(frame_num):
                frame_masks = [obj_masks[frame_idx] for obj_masks in object_mask_list]
                
                merged_mask = None
                for mask_rle in frame_masks:
                    if mask_rle is not None:
                        curr_mask = maskUtils.decode(mask_rle)
                        if merged_mask is None:
                            merged_mask = curr_mask
                        else:
                            merged_mask = np.logical_or(merged_mask, curr_mask)
                if merged_mask is None:
                    merged_mask = np.zeros((height, width), dtype=np.uint8)
                mask_list.append(merged_mask)
            null_flag = False

        if expl is not None and expl != '':
            prompt_type = PROMPT_DICT[expr_type]['with_expl']
            prompt = prompt_type.format(expression=expr)
        else:
            prompt_type = PROMPT_DICT[expr_type]['base']
            prompt = prompt_type.format(expression=expr)
        
        if '<video>\n<audio>\n' not in prompt:
            prompt = '<video>\n<audio>\n' + prompt
            
        if frame_num > self.sample_frames:
            visable_frame_index = sample_frames(frame_num, self.sample_frames)
        else:
            visable_frame_index = list(range(frame_num))
        dense_frame_index = sample_frames(len(visable_frame_index), self.dense_sample_frames)
        dense_frame_index = [visable_frame_index[i] for i in dense_frame_index]
        
        frame_flag_list = []
        for i in range(frame_num):
            if i in visable_frame_index:
                if i in dense_frame_index:
                    frame_flag_list.append(2)
                else:
                    frame_flag_list.append(1)
            else:
                frame_flag_list.append(0)
        
        visable_image_list = [image_list[i] for i in visable_frame_index]
        
        special_tokens = '\n'.join(['Frame{}: <image><sub_audio>'.format(i + 1) for i in range(len(visable_image_list))])
        prompt = prompt.replace('<video>', special_tokens)
            
        images = [Image.open(image_path).convert('RGB') for image_path in image_list]
        
        audio_list = [audio_path]
        
        if referring_speech is not None:
            audio_list.append(os.path.join(self.referring_speech_path, referring_speech))
        
        if referring_image is not None:
            assert '<image>' in expr
            images.append(Image.open(os.path.join(self.referring_image_path, referring_image)).convert('RGB'))
            frame_flag_list.append(2)

        if referring_sound is not None:
            assert '<audio>' in expr
            audio_list.append(os.path.join(self.referring_sound_path, referring_sound))
        
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'text': prompt,
            'pixel_values': pixel_values,
            'masks': mask_list,
            'hw': (frame_num, height, width),
            'split': expr_type,
            'level': level,
            'frame_flag_list': frame_flag_list,
            'audio_input': audio_list,
            'expl': expl,
            'exp_id': exp_id,
            'video_name': video_name
        }


class InferenceSampler(torch.utils.data.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    prompt = None
    random.seed(args.seed)
    summaries = dict()
    for ds_name in args.datasets:
        summaries[ds_name] = dict()

    for ds_name in args.datasets:
        ds_name_, split = ds_collections[ds_name]
        
        dataset = OmniAVSDataset(
            ds_name=ds_name_,
            split=split,
            prompt=prompt,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=1
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        for _, (pixel_values, questions, masks, hws, splits, frame_flag_lists, levels, audio_inputs, expls, exp_ids, video_names) in enumerate(tqdm(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=100,
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            frame_flag_list = frame_flag_lists[0]
            num_visable_frames = len([i for i in frame_flag_list if i != 0])
            num_patches_list = [1] * num_visable_frames
            num_image_token = []
            for i in range(len(frame_flag_list)):
                if frame_flag_list[i] == 1:
                    num_image_token.append(1)
                elif frame_flag_list[i] == 2:
                    num_image_token.append(256)
                else:
                    continue
            if len(frame_flag_list) < 200:
                pred, pred_mask = model.chat_omniavs_new(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    audio_inputs=audio_inputs[0],
                    question=questions[0],
                    generation_config=generation_config,
                    verbose=False,
                    frame_flag_list=frame_flag_lists[0],
                    num_image_token=num_image_token,
                    num_patches_list=num_patches_list
                )
            else:
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
                # Split video into chunks and process separately
                chunk_size = len(frame_flag_list) // 2  # Try processing half at a time
                
                pred_masks = []
                for i in range(0, len(frame_flag_list), chunk_size):
                    chunk_flags = frame_flag_list[i:i+chunk_size]
                    chunk_pixels = pixel_values[i:i+chunk_size]
                    
                    # Recalculate num patches and tokens for chunk
                    chunk_visable = len([f for f in chunk_flags if f != 0])
                    chunk_patches = [1] * chunk_visable
                    chunk_tokens = []
                    for f in chunk_flags:
                        if f == 1:
                            chunk_tokens.append(1)
                        elif f == 2:
                            chunk_tokens.append(256)
                        else:
                            continue
                            
                    _, chunk_mask = model.chat_omniavs_new(
                        tokenizer=tokenizer,
                        pixel_values=chunk_pixels,
                        audio_inputs=audio_inputs[0],
                        question=questions[0], 
                        generation_config=generation_config,
                        verbose=False,
                        frame_flag_list=chunk_flags,
                        num_image_token=chunk_tokens,
                        num_patches_list=chunk_patches
                    )
                    if chunk_mask is not None:
                        pred_masks.append(chunk_mask)

                # Combine results
                pred = _ # Skip text output
                try:
                    pred_mask = torch.cat(pred_masks, dim=1) if all(m is not None for m in pred_masks) else None
                except:
                    pred_mask = None
            
            print(pred, "++++++++", expls[0])

            answers = [pred]
            pred_masks = [pred_mask]

            for mask, hw, answer, pred_mask, split, level, expl, exp_id, video_name in zip(masks, hws, answers, pred_masks, splits, levels, expls, exp_ids, video_names):
                if pred_mask is not None:
                    if 'image' in dataset.split:
                        pred_mask = pred_mask.float().cpu().numpy()[:, :-1]  # remove referring mask
                    else:
                        pred_mask = pred_mask.float().cpu().numpy()
                outputs.append({
                    'answer': answer,
                    'gt_mask': mask,
                    'pred_mask': pred_mask,
                    'hw': hw,
                    'split': split,
                    'level': level,
                    'expl': expl,
                    'exp_id': exp_id,
                    'video_name': video_name
                })
        
        torch.distributed.barrier()

        metric_dict = defaultdict(list)
        exp_metric_dict = defaultdict(dict)
        try:
            for i, output in enumerate(tqdm(outputs)):
                exp_id = output['exp_id']
                video_name = output['video_name']
                if output['expl'] is not None and output['expl'] != '':
                    answer = output['answer']
                    import re
                    match = re.search(r'Explanation:\s*(.*)', answer)
                    if match:
                        pred_expl = match.group(1)
                    if answer.startswith("Sure it is"):
                        pred_expl = answer.split("Explanation:")[1].strip()
                    
                    bleu_score = BLEU(output['expl'], pred_expl)
                    metric_dict['bleu'].append(bleu_score)
                    exp_metric_dict[video_name+'_'+str(exp_id)]['bleu'] = bleu_score
                    
                    meteor_score = METEOR(output['expl'], pred_expl)
                    metric_dict['meteor'].append(meteor_score)
                    exp_metric_dict[video_name+'_'+str(exp_id)]['meteor'] = meteor_score
                
                pred_mask = output['pred_mask']

                t, h, w = output['hw']
                gt_mask = output['gt_mask']
                if pred_mask is not None:
                    pred_mask = torch.tensor(output['pred_mask'] > 0.).float()[0]
                    pred_mask = torch.nn.functional.interpolate(pred_mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)
                    pred_mask = pred_mask[:, 0] > 0.
                else:
                    pred_mask = torch.zeros((t, h, w))
                split = output['split']
                
                assert gt_mask is not None
                masks = torch.from_numpy(np.stack(gt_mask, axis=0)) > 0.
                pred_mask = pred_mask.unsqueeze(1)
                masks = masks.unsqueeze(1)
                
                if masks.sum() == 0:
                    if pred_mask.sum() == 0:
                        j = torch.tensor(1.0)
                        f = 1.0
                    else:
                        j = torch.tensor(0.0)
                        f = 0.0
                else:
                    j = mask_iou(pred_mask.float(), masks.float())
                    f = db_eval_boundary(np.array(pred_mask.cpu()[:, 0]).astype(np.uint8), np.array(masks.cpu())[:, 0].astype(np.uint8)).mean()
                
                metric_dict['j'].append(j.cpu())
                metric_dict['f'].append(f)
                exp_metric_dict[video_name+'_'+str(exp_id)  ]['j'] = j.cpu().item()
                exp_metric_dict[video_name+'_'+str(exp_id)]['f'] = f
                exp_metric_dict[video_name+'_'+str(exp_id)]['j&f'] = 0.5 * (j.cpu().item() + f)
                
                

        except Exception as e:
            print(f"Error during evaluation on rank {torch.distributed.get_rank()}: {e}")

        world_size = torch.distributed.get_world_size()
        all_metrics = [None for _ in range(world_size)]
        all_exp_metrics = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(all_metrics, metric_dict)
        torch.distributed.all_gather_object(all_exp_metrics, exp_metric_dict)

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            merged_metric_dict = defaultdict(list)
            merged_exp_metric_dict = {}
            for metrics in all_metrics:
                if metrics is not None:  # Check if metrics is not None
                    for k, v in metrics.items():
                        merged_metric_dict[k].extend(v)
            for exp_metrics in all_exp_metrics:
                if exp_metrics is not None:  # Check if exp_metrics is not None
                    merged_exp_metric_dict.update(exp_metrics)
            metric_dict = merged_metric_dict
            exp_metric_dict = merged_exp_metric_dict

            print(f'Evaluating {ds_name} ...')
            if dataset.split != 'test_n':
                print(f'{ds_name} IoU: {np.mean(metric_dict[f"j"])}, {ds_name} F: {np.mean(metric_dict[f"f"])}')
                print(f'{ds_name} BLEU: {np.mean(metric_dict[f"bleu"])}')
                print(f'{ds_name} METEOR: {np.mean(metric_dict[f"meteor"])}')
            else:
                print(f'test_n s: {np.mean(metric_dict[f"test_n"])}')
            
            if torch.distributed.get_rank() == 0:
                out_path = '_'.join(args.checkpoint.split('/')[-2:])
                writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
                if dataset.split != 'test_n':
                    summary = [
                        args.checkpoint, ds_name,
                        f'{ds_name} IoU: {np.mean(metric_dict["j"])},  F: {np.mean(metric_dict["f"])}, J&F:{0.5*(np.mean(metric_dict["j"]) + np.mean(metric_dict["f"]))}\n',
                        f'{ds_name} BLEU: {np.mean(metric_dict["bleu"])}\n',
                        f'{ds_name} METEOR: {np.mean(metric_dict["meteor"])}\n'
                    ]
                else:
                    summary = [
                        args.checkpoint, ds_name,
                        f'test_n s: {np.mean(metric_dict["test_n"])}\n'
                    ]
                print(summary)
                writer.write(' '.join(summary))
                writer.close()

                exp_writer = open(os.path.join(args.out_dir, f'{dataset.split}_{out_path}_exp_metrics.json'), 'w')
                json.dump(exp_metric_dict, exp_writer, indent=2)
                exp_writer.close()

        torch.distributed.barrier()
        
        # Collect metrics for final summary
        if ds_name in summaries:
            summaries[ds_name] = {
                'j&f': 0.5 * (np.mean(metric_dict["j"]) + np.mean(metric_dict["f"])),
                'meteor': np.mean(metric_dict["meteor"]) if "meteor" in metric_dict else 0
            }
    
    # Calculate and output final metrics
    if torch.distributed.get_rank() == 0:
        # Calculate J&F average across all 8 splits
        jf_values = [data['j&f'] for ds_name, data in summaries.items() if 'j&f' in data]
        if jf_values:
            jf_average = np.mean(jf_values)
            print(f"Average J&F across all splits: {jf_average}")
            
            # Write to summary file
            out_path = '_'.join(args.checkpoint.split('/')[-2:])
            with open(os.path.join(args.out_dir, f'{out_path}_summary.txt'), 'a') as f:
                f.write(f"Average J&F across all splits: {jf_average}\n")
        
        # Calculate METEOR average across speech and text splits
        speech_meteor = None
        text_meteor = None
        for ds_name, data in summaries.items():
            if 'speech' == ds_name and 'meteor' in data:
                speech_meteor = data['meteor']
            if 'text' == ds_name and 'meteor' in data:
                text_meteor = data['meteor']
        
        if speech_meteor is not None and text_meteor is not None:
            meteor_average = (speech_meteor + text_meteor) / 2
            print(f"Average METEOR across speech and text splits: {meteor_average}")
            
            # Write to summary file
            out_path = '_'.join(args.checkpoint.split('/')[-2:])
            with open(os.path.join(args.out_dir, f'{out_path}_summary.txt'), 'a') as f:
                f.write(f"Average METEOR across speech and text splits: {meteor_average}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='all')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results2')
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--save_mask', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    if args.datasets == 'all':
        args.datasets = ",".join(ds_collections.keys())
        
    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    PATTERN = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')
    model, tokenizer = load_model_and_tokenizer(args)
    model.seg_token_id = tokenizer.convert_tokens_to_ids('[SEG]')
    model.sub_aud_context_token_id = tokenizer.convert_tokens_to_ids('<SUB_AUDIO_CONTEXT>')
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model()
