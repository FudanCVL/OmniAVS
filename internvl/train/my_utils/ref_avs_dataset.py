import gc
import logging
import os
import os.path as osp
import random
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

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

answer_template = [
    f"Sure, it is {SEG_TOKEN}.",
]

null_answer_list = [
    "No target matches this expression."
]


def load_refavs_annotation(meta_path, split):
    # Read metadata.csv using pandas
    import pandas as pd
    
    # Read CSV file
    df = pd.read_csv(meta_path)
    # Filter for train split only
    df = df[df['split'] == split]
    
    # Convert to dictionary format
    raw_data = []
    for _, row in df.iterrows():
        video_id = row['vid']
        uid = row['uid']
        fid = row['fid']
        expression = row['exp']
        
        raw_data.append({
            'video_id': video_id,
            'uid': uid,
            'fid': fid,
            'exp': expression
        })
    
    return raw_data


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


class RefAVSDataset(RVOSDataset):
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
        
        if ds_name == 'refavs':
            self.raw_data = load_refavs_annotation(meta['meta_path'], meta['split'])
        
        print(len(self.raw_data))
        
        
        
        
        # if ds_name in ['mevis', 'lvvis', 'revos']:
        #     self.raw_data, self.mask_dict, _, _ = load_mevis_json(meta['root'], meta['expression_path'], ds_name, True)
        # elif ds_name in ['refer_ytvos']:
        #     self.raw_data, self.mask_dict, _, _ = load_refytvos_json(meta['root'], meta['expression_path'], ds_name, meta['mask_dict_path'], True)

        for data_item in self.raw_data:
            conversations = []
            question = random.choice(question_list).format(expression=data_item['exp'])
            answer = random.choice(answer_template)
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

        # 从meta中获取帧数和帧间间隔参数
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
        
        audio_path = os.path.join(self.meta['media_path'], data['video_id'], f"audio.wav")
        frames_dir = os.path.join(self.meta['media_path'], data['video_id'], "frames")
        mask_path = os.path.join(self.meta['gt_mask_path'], data['video_id'], f"fid_{data['fid']}")
              
        if not os.path.exists(mask_path):
            assert data['uid'].startswith('null'), "Mask path not found"
            null_flag = True
        else:
            null_flag = False
        
        all_frames = sorted(os.listdir(frames_dir))
        num_frames = len(all_frames)
    
        # Sample 10 frames using uniform strategy
        selected_indices = sample_frames(num_frames, self.num_frames, strategy='uniform')
        selected_indices = sorted(selected_indices)
        selected_frames = [all_frames[i] for i in selected_indices]
        
        # data['frames'] = selected_frames
        image_list = [self.load_image(os.path.join(frames_dir, frame)) for frame in selected_frames]
        
        if self.dense_frame_strategy is not None:
            dense_frame_index = sample_frames(len(image_list), self.dense_frame_num, strategy=self.dense_frame_strategy)
            dense_frame_flag_list = [1 if i in dense_frame_index else 0 for i in range(len(image_list))]
        else:
            dense_frame_flag_list = [1] * len(image_list)
        
        if null_flag:
            data['conversations'][1]['value'] = random.choice(null_answer_list)
        else:
            pass

        if '<video>' not in data['conversations'][0]['value']:
            data['conversations'][0]['value'] = '<video>\n' + '<audio>\n' + data['conversations'][0]['value']
        
        # Generate special tokens for each video frame with sub_audio
        special_tokens = '\n'.join(['Frame{}: <image><sub_audio>'.format(i + 1) for i in range(len(image_list))])
        data['conversations'][0]['value'] = data['conversations'][0]['value'].replace('<video>', special_tokens)
        
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
            if flag == 1:
                num_image_tokens.append(self.num_image_token)
            else:
                num_image_tokens.append(1)
        
        audio_info = process_audiov2(audio_path)
        audio_info_list = [audio_info]
        num_audio_tokens = [audio_info['audio_token_num']]
        audio_input_list = [torch.tensor(audio_info['input_audios'])]
        input_audio_lengths = [audio_info['input_audio_lengths']]
        
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

        mask_transform = self.get_mask_transform()
        if not null_flag:
            mask_path_list = [os.path.join(mask_path, f"{int(frame.split('.')[0]):05d}.png") for frame in selected_frames]
            masks = []
            for file_path in mask_path_list:
                image = Image.open(file_path).convert('L')  # convert to gray
                masks.append(np.array(image))
            merge_mask = [mask_transform(Image.fromarray(mask.astype(np.uint8))) for mask in masks]
            masks = torch.concat(merge_mask, dim=0)
            mask_flag = torch.ones((masks.shape[0], ), dtype=torch.long)
            if torch.all(masks == 0):
                raise ValueError(f"No valid pixel values at index {idx}")
        else:
            masks = torch.zeros((num_patches, self.image_size, self.image_size), )
            mask_flag = torch.zeros((num_patches, ), dtype=torch.long)
        
        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            name=self.ds_name,
            input_audios=input_audios,
            input_audio_lengths=input_audio_lengths,
            audio_flags=torch.tensor([2] + [1] * (len(audio_input_list) - 1), dtype=torch.long),
            dense_frame_flag=torch.tensor(dense_frame_flag_list, dtype=torch.long)
        )
        ret['image_num'] = ret['image_flags'].new_tensor([ret['image_flags'].shape[0]])
        ret['masks'] = masks
        ret['mask_flag'] = mask_flag
        ret['seg_image_num'] = (ret['dense_frame_flag']!=2).sum().unsqueeze(0)
        
        if not null_flag:
            if torch.all(masks == 0):
                raise ValueError(f"No valid pixel values at index {idx}")
        
        return ret
    