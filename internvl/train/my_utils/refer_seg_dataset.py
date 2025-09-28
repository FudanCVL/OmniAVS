import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

import glob
import json
import os
import random
import sys
import time
import pickle
import logging
import itertools
import math
import gc
from copy import deepcopy
from collections import defaultdict
from pprint import pprint

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from termcolor import colored
from transformers import CLIPImageProcessor
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm, preprocess_mpt,
                                    preprocess_phi3)
from internvl.train.my_utils.rvos_dataset import RVOSDataset
from internvl.train.constants import SEG_TOKEN

logger = logging.getLogger(__name__)

from .grefer import G_REFER
from .refer import REFER

question_list = [
    # "Please give the frame number and corresponding coordinates of the object described in the sentence when it first appears in the video.  The sentence is: <ref>{expression}</ref>"
    "Please segment the object this sentence describes in this image: <ref>{expression}</ref>"
]

answer_template = [
    f"Sure, it is {SEG_TOKEN}.",
]

null_answer_list = [
    "No target matches this expression."
]


class ReferSegDataset(RVOSDataset):
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
        
        
        if self.ds_name == "refcocog":
            splitBy = "umd"
        else:
            splitBy = "unc"

        if self.ds_name == "grefcoco":
            refer_api = G_REFER(meta['data_root'], self.ds_name, splitBy)
        else:
            refer_api = REFER(meta['data_root'], self.ds_name, splitBy)
            
        ref_ids_train = refer_api.getRefIds(split="train")
        images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
        refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

        refer_seg_ds = {}
        refer_seg_ds["images"] = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

        for item in loaded_images:
            item = item.copy()
            if self.ds_name == "refclef":
                item["file_name"] = os.path.join(
                    meta['data_root'], "images/saiapr_tc-12", item["file_name"]
                )
            else:
                item["file_name"] = os.path.join(
                    meta['data_root'], "images/mscoco/images/train2014", item["file_name"]
                )
            refer_seg_ds["images"].append(item)
        refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

        print(
            "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                self.ds_name,
                splitBy,
                len(refer_seg_ds["images"]),
                len(refer_seg_ds["annotations"]),
            )
        )
        
        img2refs = {}
        for ref in refs_train:
            image_id = ref["image_id"]
            img2refs[image_id] = img2refs.get(image_id, []) + [
                ref,
            ]
        refer_seg_ds["img2refs"] = img2refs
        
        self.refer_seg_ds = refer_seg_ds
        
        self.num_expression_per_sample = meta.get('num_expression_per_sample', 1)
        
        self.raw_data = []
        for idx, image in enumerate(self.refer_seg_ds['images']):
            data_item = {}
            data_item['image'] = image
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
            # data_item['label'] = self.labels_list[idx]
            
            self.raw_data.append(data_item)
        
        if repeat_time < 1:
            self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
        if repeat_time > 1:
            assert isinstance(repeat_time, int)
            self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)

        gc.collect()
        # self.root = meta['root']
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
          
        print(f"{self.ds_name} Total images: {len(self.raw_data)}")

    def sample_data(self, idx):
        data = deepcopy(self.raw_data[idx])
        
        image_info = self.refer_seg_ds['images'][idx]
        image_path = image_info['file_name']
        image_id = image_info['id']
        refs = self.refer_seg_ds['img2refs'][image_id]
        annotations = self.refer_seg_ds['annotations']
        
        if len(refs) == 0:
            raise ValueError(f'No reference found for image {image_id}')
        
        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        
        # Sample expressions and annotation ids
        if len(sents) >= self.num_expression_per_sample:
            sampled_inds = random.sample(range(len(sents)), self.num_expression_per_sample)
            sampled_expressions = [sents[i] for i in sampled_inds]
            sampled_ann_ids = [ann_ids[i] for i in sampled_inds]
        else:
            sampled_expressions = sents
            sampled_ann_ids = ann_ids
        
        
        conversations = []
        for expression in sampled_expressions:
            question = random.choice(question_list).format(expression=expression)
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
        
        if '<image>' not in data['conversations'][0]['value']:
            data['conversations'][0]['value'] = '<image>\n' + data['conversations'][0]['value']
        
        # Merge the image path
        # image_path = self.get_image_path(data['image'])
        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)
        
        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]
            
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = maskUtils.frPyObjects(
                                    ann["segmentation"],
                                    image_info["height"],
                                    image_info["width"],
                                )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = maskUtils.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = maskUtils.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = maskUtils.decode(rle)
            m = np.sum(
                m, axis=2
            )  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)

        masks = np.stack(masks, axis=0)
            
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
        masks = [mask_transform(Image.fromarray(mask.astype(np.uint8) * 255)) for mask in masks]
        masks = torch.concat(masks, dim=0)
        mask_flag = torch.tensor([1] * len(sampled_expressions), dtype=torch.long)

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

from internvl.train.my_utils.grefcoco import load_grefcoco_json

class GReferSegDataset(RVOSDataset):
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
        
        
        if self.ds_name == "refcocog":
            splitBy = "umd"
        else:
            splitBy = "unc"
            
        self.annotation_list = load_grefcoco_json(meta['data_root'], self.ds_name, splitBy, 'train', os.path.join(meta['data_root'], 'images/mscoco/images/train2014'))
        
        self.image2anno = defaultdict(list)
        for anno in self.annotation_list:
            self.image2anno[anno['image_id']].append(anno)
        
        self.annotation_list = list(self.image2anno.values())

        self.num_expression_per_sample = meta.get('num_expression_per_sample', 1)
        
        self.raw_data = []
        for idx, image in enumerate(self.image2anno):
            data_item = {}
            # data_item['image'] = image
            # dummy conversation
            conversations = []
            data_item['anno_list'] = self.annotation_list[idx]
            data_item['image'] = image
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
            # data_item['label'] = self.labels_list[idx]
            
            self.raw_data.append(data_item)
        
        if repeat_time < 1:
            self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
        if repeat_time > 1:
            assert isinstance(repeat_time, int)
            self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)

        gc.collect()
        # self.root = meta['root']
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
        
        idx = idx % len(self.annotation_list)
        anno_list = self.annotation_list[idx]
        image_path = anno_list[0]['file_name']
        
        height, width = anno_list[0]['height'], anno_list[0]['width']
        
        sentences = []
        anno_ids = []
        for anno in anno_list:
            sentence = anno['sentence']['raw']
            sentences.append(sentence)
            anno_ids.append(anno['annotations'])
        
        if len(sentences) == 0:
            raise ValueError(f'No sentence found for image {image_path}')
        
        # Sample expressions and annotations if more than needed
        if len(sentences) >= self.num_expression_per_sample:
            sampled_indices = self.rng.choice(len(sentences), size=self.num_expression_per_sample, replace=False)
            sampled_sentences = [sentences[i] for i in sampled_indices]
            sampled_anno_ids = [anno_ids[i] for i in sampled_indices]
        else:
            sampled_sentences = sentences
            sampled_anno_ids = anno_ids
        
        conversations = []
        mask_list = []
        mask_flag = []
        for sentence, anno_id in zip(sampled_sentences, sampled_anno_ids):
            # if len(anno_id) >= 2:
            #     print(f'Multiple annotations for {sentence}')
            question = random.choice(question_list).format(expression=sentence)
            if len(anno_id) == 1 and anno_id[0]['empty'] == True:
                answer = random.choice(null_answer_list)
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
            
            if len(anno_id) == 1 and anno_id[0]['empty'] == True:
                masks = np.zeros((height, width), dtype=np.uint8)
                mask_flag.append(0)
            else:
                _mask_list = []
                for anno in anno_id:
                    assert anno['empty'] == False
                    
                    if isinstance(anno['segmentation'][0], list):
                        rle = maskUtils.frPyObjects(anno['segmentation'], height, width)
                    else:
                        rle = anno['segmentation']
                        
                    m = maskUtils.decode(rle)
                    m = np.sum(m, axis=2)
                    m = m.astype(np.uint8)
                    _mask_list.append(m)
                
                masks = np.stack(_mask_list, axis=0)
                masks = np.max(masks, axis=0)
                mask_flag.append(1)
            mask_list.append(masks)
        
        masks = np.stack(mask_list, axis=0)
        
        data['conversations'] = conversations
        
        if '<image>' not in data['conversations'][0]['value']:
            data['conversations'][0]['value'] = '<image>\n' + data['conversations'][0]['value']
        
        # Merge the image path
        # image_path = self.get_image_path(data['image'])
        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)
        
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
        masks = [mask_transform(Image.fromarray(mask.astype(np.uint8) * 255)) for mask in masks]
        masks = torch.concat(masks, dim=0)
        mask_flag = torch.tensor(mask_flag, dtype=torch.long)
        for i, mask in enumerate(masks):
            if mask_flag[i] == 1:
                if torch.all(mask == 0):
                    raise ValueError(f"{self.ds_name}: Mask at index {i} is empty when mask_flag is 1")
            else:
                if not torch.all(mask == 0):
                    raise ValueError(f"{self.ds_name}: Mask at index {i} is not empty when mask_flag is 0")

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
