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

import logging

logger = logging.getLogger(__name__)


question_list = [
    "Please segment the object this sentence describes in this image: <ref>{expression}</ref>"
]

video_question_list = [
    "Please segment the object this sentence describes in this video: <ref>{expression}</ref>"
]

answer_template = [
    "Sure, it is [SEG].",
]

null_answer_list = [
    "No target matches this expression."
]

def uniform_interval_sample(N, K):
    # 使用 linspace 生成 K 个间隔尽量相等的点，并四舍五入为整数
    return np.round(np.linspace(0, N - 1, K)).astype(int).tolist()


def init_mapillary(base_image_dir):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir):
    with open("internvl/train/my_utils/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "images", "training"))
    )
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "images",
                "training",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_cocostuff(base_image_dir):
    cocostuff_classes = []
    with open("internvl/train/my_utils/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "cocostuff", "train2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_paco_lvis(base_image_dir):
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part


class ADE20KDataset(RVOSDataset):
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
        
        if ds_name == 'ade20k':
            self.classes_list, self.images_list, self.labels_list = init_ade20k(meta['root'])
        elif ds_name == 'cocostuff':
            self.classes_list, self.images_list, self.labels_list = init_cocostuff(meta['base_image_dir'])
            self.cocostuff_class2index = {c: i for i, c in enumerate(self.classes_list)}
        elif ds_name == 'pascal_part':
            self.classes_list, self.images_list, self.labels_list = init_pascal_part(meta['base_image_dir'])
        elif ds_name == 'paco_lvis':
            self.classes_list, self.images_list, self.labels_list = init_paco_lvis(meta['base_image_dir'])

        self.raw_data = []
        for idx, image in enumerate(self.images_list):
            data_item = {}
            data_item['image'] = image
            data_item['img_id'] = image
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
            if self.ds_name in ["ade20k", "cocostuff"]:
                data_item['label'] = self.labels_list[idx]
            
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
        self.base_image_dir = meta.get('base_image_dir', 'data/segmentation/lisa_data')
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
        print(f"{self.ds_name} Total images: {self.get_image_count()}")
        
        
    def get_image_count(self):
        return len(self.raw_data)
        
    def sample_data(self, idx):
        data = deepcopy(self.raw_data[idx])
        
        if self.pesudo_video:
            return self.sample_pseudo_video(idx)
            
        if self.ds_name in ["ade20k", "cocostuff", "mapillary"]:
            label_path = data['label']
            label = Image.open(label_path)
            label = np.array(label)
            if self.ds_name == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif self.ds_name == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255
        elif self.ds_name in ["paco_lvis", "pascal_part"]:
            class_map = self.classes_list
            img_id = data['img_id']
            coco_api = self.labels_list
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if self.ds_name == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, "vlpart", self.ds_name, file_name)
            elif self.ds_name == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)

            data['image'] = image_path
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            
            # Group annotations by category_id
            category_groups = {}
            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id not in category_groups:
                    category_groups[cat_id] = []
                category_groups[cat_id].append(ann)

            # for cat_id, group in category_groups.items():
            #     if len(group) > 1:
            #         print("123")
            #     else:
            #         print("123")
            
            if len(category_groups) == 0:
                raise ValueError(f"No annotation found for image {img_id}")
            if len(category_groups) >= self.num_expression_per_sample:
                sampled_anns = []
                category_id_sample = np.random.choice(list(category_groups.keys()), size=self.num_expression_per_sample, replace=False).tolist()
                for cat_id in category_id_sample:
                    sampled_anns.append(category_groups[cat_id])
            else:
                sampled_anns = category_groups.values()
            
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann[0]["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    if random.random() < 0.5:
                        name = obj + " " + part
                    else:
                        name = "the {} of the {}".format(part, obj)
                else:
                    name = sampled_cls
                sampled_classes.append(name)
                
            
            # if len(anns) == 0:
            #     raise ValueError(f"No annotation found for image {img_id}")
            # if len(anns) >= self.num_expression_per_sample:
            #     sampled_anns = np.random.choice(
            #         anns, size=self.num_expression_per_sample, replace=False
            #     ).tolist()
            # else:
            #     sampled_anns = anns
            
            # sampled_classes = []
            # for ann in sampled_anns:
            #     sampled_cls = class_map[ann["category_id"]]
            #     if isinstance(sampled_cls, tuple):
            #         obj, part = sampled_cls
            #         if random.random() < 0.5:
            #             name = obj + " " + part
            #         else:
            #             name = "the {} of the {}".format(part, obj)
            #     else:
            #         name = sampled_cls
            #     sampled_classes.append(name)
        
        if self.ds_name in ["ade20k", "cocostuff", "mapillary"]:
            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)
            if len(unique_label) == 0:
                raise ValueError(f"No valid label found in {label_path}")  # dataloader 会从重新采样
        
            classes = [self.classes_list[class_id] for class_id in unique_label]
            num_classes_to_sample = min(len(classes), self.num_expression_per_sample)
            sampled_classes = random.sample(classes, num_classes_to_sample)
            
            # Get a list of unused classes
            unused_classes = list(set(self.classes_list) - set(classes))
            if unused_classes:
                # Insert a random unused class
                non_existent_class = random.choice(unused_classes)
                # sampled_classes.insert(random.randint(0, len(sampled_classes)), non_existent_class)  # 不用空
        else:
            unused_classes = []
        
        conversations = []
        for class_name in sampled_classes:
            question = random.choice(question_list).format(expression=class_name)
            if class_name in unused_classes:
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

        data['conversations'] = conversations
        
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
        
        # prepare mask
        if self.ds_name in ["ade20k", "cocostuff", "mapillary"]:
            mask_list = []
            mask_flag_list = []
            for class_name in sampled_classes:
                if class_name in unused_classes:
                    mask = np.zeros_like(label)
                    mask_flag_list.append(0)
                else:
                    mask = np.zeros_like(label)
                    mask[label == self.classes_list.tolist().index(class_name)] = 1
                    mask_flag_list.append(1)
                mask_list.append(mask)
        else:
            mask_list = []
            mask_flag_list = []
            for ann in sampled_anns:
                try:
                    
                    _mask_list = [coco_api.annToMask(ann) for ann in ann]
                    mask = np.zeros_like(_mask_list[0])
                    for m in _mask_list:
                        mask = np.logical_or(mask, m).astype(np.uint8)
                    
                    mask_list.append(mask)
                    mask_flag_list.append(1)
                except Exception as e:
                    print(e)
                    raise ValueError(f"No annotation found for image {img_id}")

            # masks = np.stack(masks, axis=0)
            # masks = torch.from_numpy(masks)
            # label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        
        mask_transform = self.get_mask_transform()
        masks = [mask_transform(Image.fromarray(mask.astype(np.uint8) * 255)) for mask in mask_list]
        masks = torch.concat(masks, dim=0)
        mask_flag = torch.tensor(mask_flag_list, dtype=torch.long)
        
        for i, mask in enumerate(masks):
            if mask_flag[i] == 1:
                if torch.all(mask == 0):
                    raise ValueError(f"Mask at index {i} is empty when mask_flag is 1")

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
        
        if self.ds_name in ["ade20k", "cocostuff", "mapillary"]:
            label_path = data['label']
            label = Image.open(label_path)
            label = np.array(label)
            if self.ds_name == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif self.ds_name == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255
        elif self.ds_name in ["paco_lvis", "pascal_part"]:
            class_map = self.classes_list
            img_id = self.images_list[idx]
            coco_api = self.labels_list
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if self.ds_name == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, "vlpart", self.ds_name, file_name)
            elif self.ds_name == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)

            data['image'] = image_path
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if len(anns) == 0:
                raise ValueError(f"No annotation found for image {img_id}")
            if len(anns) >= self.num_expression_per_sample:
                sampled_anns = np.random.choice(
                    anns, size=self.num_expression_per_sample, replace=False
                ).tolist()
            else:
                sampled_anns = anns
            
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    if random.random() < 0.5:
                        name = obj + " " + part
                    else:
                        name = "the {} of the {}".format(part, obj)
                else:
                    name = sampled_cls
                sampled_classes.append(name)
        
        if self.ds_name in ["ade20k", "cocostuff", "mapillary"]:
            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)
            if len(unique_label) == 0:
                raise ValueError(f"No valid label found in {label_path}")  # dataloader 会从重新采样
        
            classes = [self.classes_list[class_id] for class_id in unique_label]
            num_classes_to_sample = min(len(classes), self.num_expression_per_sample)
            sampled_classes = random.sample(classes, num_classes_to_sample)
            
            # Get a list of unused classes
            unused_classes = list(set(self.classes_list) - set(classes))
            # if unused_classes:
            #     # Insert a random unused class
            #     non_existent_class = random.choice(unused_classes)
            #     sampled_classes.insert(random.randint(0, len(sampled_classes)), non_existent_class)
        else:
            unused_classes = []
        
        conversations = []
        for class_name in sampled_classes:
            question = random.choice(video_question_list).format(expression=class_name)
            if class_name in unused_classes:
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

        data['conversations'] = conversations
        
        if '<video>' not in data['conversations'][0]['value']:
            data['conversations'][0]['value'] = '<video>\n' + data['conversations'][0]['value']
        
        transform = self.get_transform()
        image = self.load_image(data['image'])
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
        
        # prepare mask
        if self.ds_name in ["ade20k", "cocostuff", "mapillary"]:
            mask_list = []
            mask_flag_list = []
            for class_name in sampled_classes:
                if class_name in unused_classes:
                    mask = np.zeros_like(label)
                    mask_flag_list.append(0)
                else:
                    mask = np.zeros_like(label)
                    mask[label == self.classes_list.tolist().index(class_name)] = 1
                    mask_flag_list.append(1)
                mask_list.append(mask)
        else:
            mask_list = []
            mask_flag_list = []
            for ann in sampled_anns:
                try:
                    mask_list.append(coco_api.annToMask(ann))
                    mask_flag_list.append(1)
                except Exception as e:
                    print(e)
                    raise ValueError(f"No annotation found for image {img_id}")
        
        mask_transform = self.get_mask_transform()
        masks = [mask_transform(Image.fromarray(mask.astype(np.uint8) * 255)) for mask in mask_list]
        masks = torch.concat(masks * self.dense_frame_num, dim=0)
        mask_flag = torch.tensor(mask_flag_list * self.dense_frame_num, dtype=torch.long)
        
        for i, mask in enumerate(masks):
            if mask_flag[i] == 1:
                if torch.all(mask == 0):
                    raise ValueError(f"Mask at index {i} is empty when mask_flag is 1")

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
            seg_image_num = torch.tensor([self.dense_frame_num], dtype=torch.long)
        )
        ret['image_num'] = ret['image_flags'].new_tensor([ret['image_flags'].shape[0]])
        
        return ret

if __name__ == '__main__':
    init_ade20k("data/segmentation")
    