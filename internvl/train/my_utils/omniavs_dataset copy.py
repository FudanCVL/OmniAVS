import gc
import logging
import os
import os.path as osp
import random
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import json

# Third party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
from termcolor import colored
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPImageProcessor

# Local imports
from .d2_datasets.mevis_utils import load_mevis_json
from .d2_datasets.refytvos_utils import load_refytvos_json
from internvl.train.constants import SEG_TOKEN
from internvl.train.dataset import (
    ConcatDataset,
    TCSLoader,
    WeightedConcatDataset,
    build_transform,
    dynamic_preprocess,
    preprocess,
    preprocess_internlm,
    preprocess_mpt,
    preprocess_phi3
)
from internvl.train.my_utils.rvos_dataset import RVOSDataset
from internvl.model.qwen_audio.audio import process_audiov2


logger = logging.getLogger(__name__)


question_list = [
    "Please segment the object this sentence describes in this video: <ref>{expression}</ref>"
]

question_list_speech = [
    "Please segment the object this sentence describes in this video: <ref><audio></ref>"
]

question_list_speech_sound = [
    "Please segment the object this sentence describes in this video: <ref><audio></ref>, referring audio is <audio>."
]

question_list_speech_image = [
    "Please segment the object this sentence describes in this video: <ref><audio></ref>, referring image is <image>"
]

question_list_speech_sound_image = [
    "Please segment the object this sentence describes in this video: <ref><audio></ref>, referring audio is <audio>, referring image is <image>"
]

question_list_speech_with_expl = [
    "Please segment the object this sentence describes in this video: <ref><audio></ref>, and explain why."
]

question_list_speech_sound_with_expl = [
    "Please segment the object this sentence describes in this video: <ref><audio></ref>, referring audio is <audio>, and explain why."
]

question_list_speech_image_with_expl = [
    "Please segment the object this sentence describes in this video: <ref><audio></ref>, referring image is <image>, and explain why."
]

question_list_speech_sound_image_with_expl = [
    "Please segment the object this sentence describes in this video: <ref><audio></ref>, referring audio is <audio>, referring image is <image>, and explain why."
]

question_list_with_expl = [
    "Please segment the object this sentence describes in this video: <ref>{expression}</ref>, and explain why."
]

answer_list_with_expl = [
    f"Sure, it is {SEG_TOKEN}" + ". Explanation: {explanation}",
]

answer_list = [
    f"Sure, it is {SEG_TOKEN}."
]

null_answer_list = [
    "No target matches this expression."
]

null_answer_list_with_expl = [
    "No target matches this expression." + ". Explanation: {explanation}"
]


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


class OmniAVSDataset(RVOSDataset):
    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=224,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        min_num_frame=4,  # for video data
        max_num_frame=12,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        random_seed=0,
        qwa_version='v1'
    ):
        super(RVOSDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method
        
        # Load meta expressions from jsonl file
        self.raw_data = []
        with open(meta['meta_expression_path'], 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.raw_data.append(data)
        
        self.video_dict = json.load(open(meta['video_dict_path'], 'r'))
        
        self.root = meta['root']
        self.image_path = osp.join(self.root, 'images')
        self.audio_path = osp.join(self.root, 'audios')
        self.referring_image_path = osp.join(self.root, 'referrings', 'referring_images')
        self.referring_speech_path = osp.join(self.root, 'referrings', 'referring_speeches')
        self.referring_sound_path = osp.join(self.root, 'referrings', 'referring_sounds')

        for data_item in self.raw_data:
            conversations = []
            question = random.choice(question_list).format(expression=data_item['expr'])
            answer = random.choice(answer_list)
            conversations.append({
                "from": "human",
                "value": question
            })
            conversations.append({
                "from": "gpt",
                "value": answer
            })
            data_item['conversations'] = conversations
        

        if repeat_time < 1:
            self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
        if repeat_time > 1:
            assert isinstance(repeat_time, int)
            self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)

        gc.collect()
        # self.root = meta['root']
        self.meta = meta
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        self.qwa_version = qwa_version

        self.num_frames = meta.get('num_frames', 10)
        self.max_frame_interval = meta.get('max_frame_interval', 5)
        self.dense_frame_num = meta.get('dense_frame_num', 4)
        self.dense_frame_strategy = meta.get('dense_frame_strategy', 'uniform')
        if self.group_by_length:
            self.conv2length = {}
            self.length = []
            for data_item in self.raw_data:
                conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                str_length = len(conversations)

                if str_length not in self.conv2length:
                    token_length = tokenizer(
                        conversations, return_tensors='pt', padding=False, truncation=False,
                    ).input_ids.size(1)
                    self.conv2length[str_length] = token_length + num_image_token * (
                                max_dynamic_patch + use_thumbnail)
                else:
                    token_length = self.conv2length[str_length]
                
                self.length.append(token_length)
        gc.collect()
    
    
    def sample_data(self, idx):
        data = deepcopy(self.raw_data[idx])
        
        video_name = data['video_name']
        
        frames_dir = osp.join(self.image_path, video_name)
        audio_path = osp.join(self.audio_path, video_name + '.mp3')
        video_info = self.video_dict[video_name]
        frame_num = len(video_info['file_name'])
        obj_id = data['obj_id']
        expr_type = data['expr_type']
        assert expr_type in [
            "text", "speech",
            "text_sound", "speech_sound",
            "text_image", "speech_image",
            "text_sound_image", "speech_sound_image"
        ]
        
        expr = data['expr']
        expl = data['expl']
    
            
        # set the answer
        if len(obj_id) == 0:
            if expl == None or expl == "":
                answer = random.choice(null_answer_list)
            else:
                answer = random.choice(null_answer_list_with_expl).format(explanation=expl)
        else:
            if expl == None or expl == "":
                answer = random.choice(answer_list).format(expression=expr)
            else:
                answer = random.choice(answer_list_with_expl).format(expression=expr, explanation=expl)
        data['conversations'][1]['value'] = answer
    
        # sample frame
        video_duration = video_info['duration']
        # Sample frames uniformly, always including first and last frames
        selected_indices = sample_frames(frame_num, self.num_frames, strategy='uniform')
        selected_indices = sorted(selected_indices)
        
        image_list = []
        for selected_index in selected_indices:
            image_list.append(self.load_image(os.path.join(frames_dir, video_info['file_name'][selected_index])))
        
        # get_mask
        if len(obj_id) == 0:
            null_flag = True
            mask_list = [np.zeros((self.image_size, self.image_size), dtype=np.uint8)] * len(selected_indices)
            mask_flag = torch.zeros((len(selected_indices), ), dtype=torch.long)
        else:
            object_mask_list = []
            for i in range(len(obj_id)):
                mask_dict = video_info["mask_dict"][str(obj_id[i])]
                selected_masks = [mask_dict[selected_index] for selected_index in selected_indices]
                object_mask_list.append(selected_masks)
            
            # Merge masks for each frame
            mask_list = []
            for frame_idx in range(len(selected_indices)):
                # Get masks for all objects in current frame
                frame_masks = [obj_masks[frame_idx] for obj_masks in object_mask_list]
                
                # Decode RLE masks and merge
                merged_mask = None
                for mask_rle in frame_masks:
                    if mask_rle is not None:
                        # Decode RLE to binary mask
                        curr_mask = maskUtils.decode(mask_rle)
                        if merged_mask is None:
                            merged_mask = curr_mask
                        else:
                            # Merge with existing mask using logical OR
                            merged_mask = np.logical_or(merged_mask, curr_mask)
                if merged_mask is None:
                    merged_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
                mask_list.append(merged_mask)
            mask_flag = torch.ones((len(selected_indices), ), dtype=torch.long)
            null_flag = False
        mask_transform = self.get_mask_transform()
        masks = [mask_transform(Image.fromarray(mask.astype(np.uint8)*255)) for mask in mask_list]
        masks = torch.concat(masks, dim=0)
        # # data['frames'] = selected_frames
        # image_list = [self.load_image(os.path.join(frames_dir, frame)) for frame in selected_frames]
        
        self.audio_interleaving = True
        if self.dense_frame_strategy is not None:
            dense_frame_index = sample_frames(len(image_list), self.dense_frame_num, strategy=self.dense_frame_strategy)
            dense_frame_flag_list = [1 if i in dense_frame_index else 0 for i in range(len(image_list))]
        else:
            dense_frame_flag_list = [1] * len(image_list)
            
        # Generate special tokens for each video frame
        
        referring_image_path = data['referring_image']
        referring_speech_path = data['referring_speech']
        referring_sound_path = data['referring_sound']
        
        audio_list = [audio_path]
        if referring_speech_path is not None:
            audio_list.append(os.path.join(self.referring_speech_path, referring_speech_path))
            if expr_type == "speech":
                if expl == None or expl == "":
                    data['conversations'][0]['value'] = random.choice(question_list_speech)
                else:
                    data['conversations'][0]['value'] = random.choice(question_list_speech_with_expl)
            elif expr_type == "speech_sound":
                if expl == None or expl == "":
                    data['conversations'][0]['value'] = random.choice(question_list_speech_sound)
                else:
                    data['conversations'][0]['value'] = random.choice(question_list_speech_sound_with_expl)
            elif expr_type == "speech_image":
                if expl == None or expl == "":
                    data['conversations'][0]['value'] = random.choice(question_list_speech_image)
                else:
                    data['conversations'][0]['value'] = random.choice(question_list_speech_image_with_expl)
            elif expr_type == "speech_sound_image":
                if expl == None or expl == "":
                        data['conversations'][0]['value'] = random.choice(question_list_speech_sound_image)
                else:
                    data['conversations'][0]['value'] = random.choice(question_list_speech_sound_image_with_expl)
            else:
                raise ValueError(f"Invalid expr_type: {expr_type}")
        
        if '<video>' not in data['conversations'][0]['value']:
            data['conversations'][0]['value'] = '<video>\n' + '<audio>\n' + data['conversations'][0]['value']
        special_tokens = '\n'.join(['Frame{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data['conversations'][0]['value'] = data['conversations'][0]['value'].replace('<video>', special_tokens)
        
        if referring_image_path is not None:
            assert '<image>' in expr
            image_list.append(self.load_image(os.path.join(self.referring_image_path, referring_image_path)))
            dense_frame_flag_list.append(2)

        if referring_sound_path is not None:
            assert '<audio>' in expr
            audio_list.append(os.path.join(self.referring_sound_path, referring_sound_path))

        transform = self.get_transform()
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        
        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        # num_image_tokens = [self.num_image_token] * num_patches
        num_image_tokens = []
        for flag in dense_frame_flag_list:
            if flag == 1 or flag == 2:
                num_image_tokens.append(self.num_image_token)
            else:
                num_image_tokens.append(1)
        
        audio_info_list = []
        num_audio_tokens = []
        audio_input_list = []
        input_audio_lengths = []
        for audio_path in audio_list:
            audio_info = process_audiov2(audio_path)
            audio_info_list.append(audio_info)
            num_audio_tokens.append(audio_info['audio_token_num'])
            audio_input_list.append(torch.tensor(audio_info['input_audios']))
            input_audio_lengths.append(audio_info['input_audio_lengths'])
        
        input_audios = torch.stack(audio_input_list, dim=0)
        input_audio_lengths = torch.stack(input_audio_lengths, dim=0)
        
        ret = preprocess_function(self.template_name, 
                                  [deepcopy(data['conversations'])],
                                  self.tokenizer, 
                                  num_image_tokens, 
                                  group_by_length=self.group_by_length,
                                  ds_name=self.ds_name, 
                                  num_image=num_patches,
                                  audio_enable=True,
                                  num_audio_token=num_audio_tokens)
        
        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * len(image_list), dtype=torch.long),
            name=self.ds_name,
            input_audios=input_audios,
            input_audio_lengths=input_audio_lengths,
            audio_flags=torch.tensor([1] * len(audio_list), dtype=torch.long),
            dense_frame_flag=torch.tensor(dense_frame_flag_list, dtype=torch.long)
        )
        ret['image_num'] = ret['image_flags'].new_tensor([ret['image_flags'].shape[0]])
        ret['seg_image_num'] = (ret['dense_frame_flag']!=2).sum().unsqueeze(0)
        ret['masks'] = masks
        ret['mask_flag'] = mask_flag
        
        if not null_flag:
            if torch.all(masks == 0):
                raise ValueError(f"No valid pixel values at index {idx}")
        
        return ret
    