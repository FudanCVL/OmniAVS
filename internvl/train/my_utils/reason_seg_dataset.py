import json
import os
import random
import gc
from copy import deepcopy
import cv2
import numpy as np
import torch
import glob

from PIL import Image
import torchvision.transforms as T

from internvl.train.dataset import dynamic_preprocess
from internvl.train.my_utils.rvos_dataset import RVOSDataset

import logging

logger = logging.getLogger(__name__)


question_list = [
    "Please segment the object this sentence describes in this image: <ref>{expression}</ref>"
]

# long_question_list = [
#     "{Express}"
# ]

video_question_list = [
    "Please segment the object this sentence describes in this video: <ref>{expression}</ref>"
]

question_with_explanation = [
    "Please segment the object this sentence describes in this image: <ref>{expression}</ref>, and explain why."
]

answer_template = [
    "Sure, it is [SEG].",
]

answer_template_with_explanation = [
    "Sure, it is [SEG]. Explanation: {explanation}"
]

null_answer_list = [
    "No target matches this expression."
]


def uniform_interval_sample(N, K):
    return np.round(np.linspace(0, N - 1, K)).astype(int).tolist()


def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]
    img = cv2.imread(img)
    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 1  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence


class ReasonSegDataset(RVOSDataset):
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
        
        self.num_expression_per_sample = meta.get('num_expression_per_sample', 1)
        self.num_frames = meta.get('num_frames', 36)
        self.dense_frame_num = meta.get('dense_frame_num', 4)
        self.pesudo_video = meta.get('pesudo_video', False)
        self.images_list = glob.glob(os.path.join(meta['root'], "*.jpg"))
        self.jsons_list = [path.replace(".jpg", ".json") for path in self.images_list]
        self.explanations_list = json.load(open("playground/data/segmentation/reason_seg/explanatory/train.json"))
        self.img_to_explanation = {}
        for item in self.explanations_list:
            self.img_to_explanation[item['image']] = item
        
        self.raw_data = []
        for idx, image in enumerate(self.images_list):
            data_item = {}
            data_item['image'] = image
            data_item['json'] = self.jsons_list[idx]
            # dummy conversation
            conversations = []
            question = random.choice(question_list).format(expression='dummy')
            answer = random.choice(answer_template)
            conversations.append({
                "from": "human",
                "value": "<image>\n" + question
            })
            conversations.append({
                "from": "gpt",
                "value": answer
            })
            data_item['conversations'] = conversations
            
            self.raw_data.append(data_item)
        

        if repeat_time < 1:
            self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
        if repeat_time > 1:
            assert isinstance(repeat_time, int)
            self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)

        gc.collect()
        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        self.qwa_version = qwa_version

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
        # idx = idx % len(self.jsons_list)
        
        if self.pesudo_video:
            return self.sample_pseudo_video(idx)
        
        json_path = data['json']
        image_path = data['image']
        json_info = json.load(open(json_path))
        
        mask, comments, is_sentence = get_mask_from_json(json_path, image_path)
        
        image_name = os.path.basename(image_path)
        if image_name in self.img_to_explanation:
            explanation = self.img_to_explanation[image_name]['outputs']
        else:
            explanation = None
        
        # Sample expressions from comments
        if len(comments) >= self.num_expression_per_sample:
            sampled_expressions = random.sample(comments, self.num_expression_per_sample)
        else:
            sampled_expressions = comments
        
        conversations = []
        # explanation = None
        for expression in sampled_expressions:
            flag = True
            if explanation:
                if random.random() < 0.5:
                    question = random.choice(question_with_explanation).format(expression=expression)
                    flag = False
                else:
                    question = random.choice(question_list).format(expression=expression)
            else:
                question = random.choice(question_list).format(expression=expression)
            
            if not flag:
                answer = random.choice(answer_template_with_explanation).format(expression=expression, explanation=explanation)
            else:
                answer = random.choice(answer_template)
                
            conversations.append({
                "from": "human",
                "value": question
            })
            conversations.append({
                "from": "gpt",
                "value": answer
            })

        data['conversations'] = conversations
        
        # if self.pesudo_video:
        #     if '<video>' not in data['conversations'][0]['value']:
        #         data['conversations'][0]['value'] = '<video>\n' + data['conversations'][0]['value']
                
        # else:
        if '<image>' not in data['conversations'][0]['value']:
            data['conversations'][0]['value'] = '<image>\n' + data['conversations'][0]['value']
        
        # Merge the image path
        # image_path = self.get_image_path(data['image'])
        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(data['image'])
        
        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]
            
        transform = self.get_transform()
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)
        
        
        mask_transform = self.get_mask_transform()
        masks = [mask_transform(Image.fromarray(mask.astype(np.uint8) * 255)) for mask in [mask] * len(sampled_expressions)]
        masks = torch.concat(masks, dim=0)
        mask_flag = torch.tensor([1] * len(sampled_expressions), dtype=torch.long)
        
        for i, mask in enumerate(masks):
            if mask_flag[i] == 1:
                if torch.all(mask == 0):
                    raise ValueError(f"Mask at index {i} is empty when mask_flag is 1")
            else:
                if not torch.all(mask == 0):
                    raise ValueError(f"Mask at index {i} is not empty when mask_flag is 0")

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            masks=masks,
            mask_flag=mask_flag,
            name=self.ds_name,
            dense_frame_flag = torch.tensor([1] * num_patches, dtype=torch.long),
            seg_image_num = torch.tensor([num_patches], dtype=torch.long)
        )
        ret['image_num'] = ret['image_flags'].new_tensor([ret['image_flags'].shape[0]])
        
        return ret

    def sample_pseudo_video(self, idx):
        data = deepcopy(self.raw_data[idx])
        json_path = data['json']
        image_path = data['image']
        json_info = json.load(open(json_path))
        
        mask, comments, is_sentence = get_mask_from_json(json_path, image_path)
        
        image_name = os.path.basename(image_path)
        if image_name in self.img_to_explanation:
            explanation = self.img_to_explanation[image_name]['outputs']
        else:
            explanation = None
        
        # Sample expressions from comments
        if len(comments) >= self.num_expression_per_sample:
            sampled_expressions = random.sample(comments, self.num_expression_per_sample)
        else:
            sampled_expressions = comments
        
        conversations = []
        explanation = None
        for expression in sampled_expressions:
            flag = True
            if explanation:
                if random.random() < 0.5:
                    question = random.choice(video_question_list).format(expression=expression)
                    flag = False
                else:
                    question = random.choice(video_question_list).format(expression=expression)
            else:
                question = random.choice(video_question_list).format(expression=expression)
            
            if not flag:
                answer = random.choice(answer_template_with_explanation).format(expression=expression, explanation=explanation)
            else:
                answer = random.choice(answer_template)
                
            conversations.append({
                "from": "human",
                "value": question
            })
            conversations.append({
                "from": "gpt",
                "value": answer
            })

        data['conversations'] = conversations
        
        if '<video>' not in data['conversations'][0]['value']:
            data['conversations'][0]['value'] = '<video>\n' + data['conversations'][0]['value']
        
        
        transform = self.get_transform()
        image = self.load_image(image_path)
        image_list = [image] * self.num_frames
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        
        special_tokens = '\n'.join(['Frame{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data['conversations'][0]['value'] = data['conversations'][0]['value'].replace('<video>\n', special_tokens)
        
        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()
        


        # Preprocess the conversations and generate the return dictionary
        dense_frame_flag = []
        # num_image_tokens = [self.num_image_token] * num_patches
        num_image_tokens = []
        dense_frame_indices = uniform_interval_sample(self.num_frames, self.dense_frame_num)
        for frame_idx in range(self.num_frames):
            if frame_idx in dense_frame_indices:
                num_image_tokens.append(self.num_image_token)
                dense_frame_flag.append(1)
            else:
                num_image_tokens.append(1)
                dense_frame_flag.append(0)
                
        ret = preprocess_function(self.template_name, [deepcopy(data['conversations'])],
                    self.tokenizer, num_image_tokens,
                    group_by_length=self.group_by_length, ds_name=self.ds_name, num_image=self.num_frames)
        
        mask_transform = self.get_mask_transform()
        masks = [mask_transform(Image.fromarray(mask.astype(np.uint8) * 255))] * self.dense_frame_num
        masks = torch.concat(masks, dim=0)
        mask_flag = torch.tensor([1] * self.dense_frame_num, dtype=torch.long)
        
        
        for i, mask in enumerate(masks):
            if mask_flag[i] == 1:
                if torch.all(mask == 0):
                    raise ValueError(f"Mask at index {i} is empty when mask_flag is 1")
            else:
                if not torch.all(mask == 0):
                    raise ValueError(f"Mask at index {i} is not empty when mask_flag is 0")

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            masks=masks,
            mask_flag=mask_flag,
            name=self.ds_name,
            dense_frame_flag = torch.tensor(dense_frame_flag, dtype=torch.long),
            seg_image_num = torch.tensor([self.dense_frame_num], dtype=torch.long),
        )
        ret['image_num'] = ret['image_flags'].new_tensor([ret['image_flags'].shape[0]])
        
        return ret
        
        
        
        
        