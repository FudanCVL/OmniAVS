###########################################################################
# Created by: BUAA
# Email: clyanhh@gmail.com
# Copyright (c) 2024
###########################################################################
import itertools
import json
import logging
import os
import os.path as osp
import pickle
import sys
import cv2
import time
import random
import torch
import math
import torch.nn.functional as F
from pprint import pprint
from termcolor import colored
from collections import defaultdict
import gc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import pycocotools.mask as maskUtils
from transformers import CLIPImageProcessor
from PIL import Image
from copy import deepcopy

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from .d2_datasets.refytvos_utils import load_refytvos_json
from .d2_datasets.mevis_utils import load_mevis_json

from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm, preprocess_mpt,
                                    preprocess_phi3)
from internvl.model.qwen_audio.audio import process_audiov2
from internvl.train.constants import SEG_TOKEN


logger = logging.getLogger(__name__)

question_list = [
    # "Please give the frame number and corresponding coordinates of the object described in the sentence when it first appears in the video.  The sentence is: <ref>{expression}</ref>"
    "Please segment the object this sentence describes in this video: <ref>{expression}</ref>"
]

answer_template = [
    f"Sure, it is {SEG_TOKEN}.",
]

null_answer_list = [
    "No target matches this expression."
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


def uniform_interval_sample(N, K):
    # 使用 linspace 生成 K 个间隔尽量相等的点，并四舍五入为整数
    return np.round(np.linspace(0, N - 1, K)).astype(int).tolist()


class RVOSDataset:
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
        if ds_name in ['mevis', 'lvvis', 'revos']:
            self.raw_data, self.mask_dict, _, _ = load_mevis_json(meta['root'], meta['expression_path'], ds_name, True)
        elif ds_name in ['refer_ytvos', 'refer_davis']:
            self.raw_data, self.mask_dict, _, _ = load_refytvos_json(meta['root'], meta['expression_path'], ds_name, meta['mask_dict_path'], True)

        for data_item in self.raw_data:
            conversations = []
            question = random.choice(question_list).format(expression=data_item['exp'])
            answer = random.choice(answer_template)
            conversations.append({
                "from": "human",
                "value": "<video>\n" + question
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

        self.num_frames = meta.get('num_frames', 10)
        self.max_frame_interval = meta.get('max_frame_interval', 5)

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
    
    def __len__(self):
        return len(self.raw_data)

    def get_transform(self):
        # Build transformation function
        
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform
    
    def get_mask_transform(self):
        transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST),
            T.ToTensor()
        ])
        return transform
    
    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')
    
    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path
    
    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        return preprocess_function
    
    def sample_data(self, idx):
        data = deepcopy(self.raw_data[idx])
        all_frames = data['frames']
        num_frames = len(all_frames)
        # selected_indices = [0]
        
        # if self.ds_name == 'refer_davis':
        #     selected_indices = sample_frames(num_frames, self.num_frames, strategy='uniform')
        #     selected_frames = [all_frames[i] for i in selected_indices]
        #     dense_frame_index = sample_frames(len(selected_frames), 4, strategy='uniform')
        #     dense_frame_flag_list = [1 if i in dense_frame_index else 0 for i in range(len(selected_frames))]
            
            
            
        # if self.dense_frame_strategy is not None:
        #     dense_frame_index = sample_frames(len(image_list), self.dense_frame_num, strategy=self.dense_frame_strategy)
        #     dense_frame_flag_list = [1 if i in dense_frame_index else 0 for i in range(len(image_list))]
        # else:
        #     dense_frame_flag_list = [1] * len(image_list)
        
        selected_indices = uniform_interval_sample(num_frames, self.num_frames)
        dense_frame_id = uniform_interval_sample(len(selected_indices), 4)
        dense_frame_indices = [selected_indices[i] for i in dense_frame_id]
        
        # print("+++++++++++++++++++", len(selected_indices))
        selected_indices.sort()
        selected_frames = [all_frames[i] for i in selected_indices]
        # data['frames'] = selected_frames
        image_list = [self.load_image(os.path.join(self.root, 'JPEGImages', data['video'], f"{frame}.jpg")) for frame in selected_frames]
        
        
        if '<video>' not in data['conversations'][0]['value']:
            data['conversations'][0]['value'] = '<video>\n' + data['conversations'][0]['value']
        
        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data['conversations'][0]['value'] = data['conversations'][0]['value'].replace('<video>\n', special_tokens)
        
        transform = self.get_transform()
        
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        
        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        dense_frame_flag = []
        # num_image_tokens = [self.num_image_token] * num_patches
        num_image_tokens = []
        for frame_idx in selected_indices:
            if frame_idx in dense_frame_indices:
                num_image_tokens.append(self.num_image_token)
                dense_frame_flag.append(1)
            else:
                num_image_tokens.append(1)
                dense_frame_flag.append(0)
        ret = preprocess_function(self.template_name, 
                                  [deepcopy(data['conversations'])],
                                  self.tokenizer, 
                                  num_image_tokens, 
                                  group_by_length=self.group_by_length,
                                  ds_name=self.ds_name, 
                                  num_image=num_patches)

        # get_mask
        mask_list = defaultdict(list)
        for anno_id in data['anno_id']:
            mask = self.mask_dict[anno_id]
            assert len(mask) == len(data['frames']), "mask length not match"        
            selected_mask = [mask[i] for i in dense_frame_indices]
            
            for j, selected_mask in enumerate(selected_mask):
                mask_list[j].append(selected_mask)
        
        # Merge masks for each frame
        merge_mask_list = []
        for frame_idx in range(len(selected_indices)):
            if frame_idx in mask_list:
                # Combine all masks for this frame
                # combined_mask = np.zeros((image_list[0].size[1], image_list[0].size[0]))
                combined_mask = None
                for segm in mask_list[frame_idx]:
                    if segm is not None:
                        m = maskUtils.decode(segm)
                        if m.ndim == 3:
                            m = m.sum(axis=2) > 0
                        else:
                            m = m > 0
                    else:
                        # raise ValueError("Mask is None")
                        continue
                    if combined_mask is None:
                        combined_mask = m
                    else:
                        combined_mask = np.logical_or(combined_mask, m)
                merge_mask_list.append(combined_mask)
            else:
                # If no mask for this frame, append an empty mask
                # merge_mask_list.append(np.zeros((image_list[0].size[1], image_list[0].size[0])))
                merge_mask_list.append(None)
        assert len(merge_mask_list) == len(selected_indices), "Merged mask list length does not match selected frames"
        
        # Check if all masks are None
        if all(mask is None for mask in merge_mask_list):
            # Return empty masks and flags if all masks are None
            data['conversations'][1]['value'] = random.choice(null_answer_list)
            ret = preprocess_function(self.template_name, 
                            [deepcopy(data['conversations'])],
                            self.tokenizer, 
                            num_image_tokens, 
                            group_by_length=self.group_by_length,
                            ds_name=self.ds_name, 
                            num_image=num_patches)
            return {
                'input_ids': ret['input_ids'][0],
                'labels': ret['labels'][0], 
                'attention_mask': ret['attention_mask'][0],
                'pixel_values': pixel_values,
                'image_flags': torch.tensor([1] * num_patches, dtype=torch.long),
                'masks': torch.zeros((len(merge_mask_list), self.image_size, self.image_size)),
                'mask_flag': torch.zeros((len(merge_mask_list),), dtype=torch.long),
                'name': self.ds_name,
                'image_num': torch.tensor([num_patches]),
                'dense_frame_flag': torch.tensor(dense_frame_flag, dtype=torch.long),
                'seg_image_num': (torch.tensor(dense_frame_flag, dtype=torch.long)==1).sum().unsqueeze(0)
            }
        else:
            # Replace None masks with zero masks
            for i in range(len(merge_mask_list)):
                if merge_mask_list[i] is None:
                    merge_mask_list[i] = np.zeros((image_list[0].size[1], image_list[0].size[0]))
        
        mask_transform = self.get_mask_transform()
        merge_mask = [mask_transform(Image.fromarray(mask.astype(np.uint8) * 255)) for mask in merge_mask_list]
        masks = torch.concat(merge_mask, dim=0)
        mask_flag = torch.ones((masks.shape[0], ), dtype=torch.long)
        
        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            name=self.ds_name
        )
        ret['image_num'] = ret['image_flags'].new_tensor([ret['image_flags'].shape[0]])
        ret['masks'] = masks
        ret['mask_flag'] = mask_flag
        ret['dense_frame_flag'] = torch.tensor(dense_frame_flag, dtype=torch.long)
        ret['seg_image_num'] = (ret['dense_frame_flag']==1).sum().unsqueeze(0)
        # ret = parse_mask_annotation(ret, data_item)  # 用来处理有没有mask的
        
        if torch.all(masks == 0):
            raise ValueError(f"No valid pixel values at index {idx}")
        return ret
    
    def __getitem__(self, idx):
        idx = idx % len(self.raw_data)
        while True:
            try:
                ret = self.sample_data(idx)
                if ret.get("input_audios", None) is None:
                    
                    # Use empty dummy audio file
                    audio_path = "playground/empty_dummy.mp3"
                    audio_info = process_audiov2(audio_path)
                    ret['audio_flags'] = torch.tensor([0], dtype=torch.long)
                    ret["input_audios"] = torch.tensor(audio_info['input_audios']).unsqueeze(0)
                    ret["input_audio_lengths"] = torch.tensor(audio_info['input_audio_lengths']).unsqueeze(0)
                break
            except ValueError as e:
                # Handle data format errors
                import traceback
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")
                logger.warning(f"{self.ds_name}: Data format error in sample {idx}")
                logger.warning(f"Error location: {e.__traceback__.tb_frame.f_code.co_name}")
                logger.warning(f"Error line: {e.__traceback__.tb_lineno}")
                logger.warning(f"Error message: {str(e)}")
                idx = random.randint(0, len(self.raw_data) - 1)
            except Exception as e:
                import traceback
                logger.warning(f"{self.ds_name}: Unexpected error in sample {idx}")
                logger.warning(f"Error location: {e.__traceback__.tb_frame.f_code.co_name}")
                logger.warning(f"Error line: {e.__traceback__.tb_lineno}")
                logger.warning(f"Error message: {str(e)}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")
                idx = random.randint(0, len(self.raw_data) - 1)
        
        return ret
            