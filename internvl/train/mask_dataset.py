import os
import json

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

MASK_SIZE = 448


def parse_mask_annotation(ret, data_item):
    dataset_name = data_item.get('dataset_name', None)
    if not dataset_name:
        masks = torch.zeros((ret['image_flags'].shape[0], MASK_SIZE, MASK_SIZE))
        mask_flag = torch.zeros((ret['image_flags'].shape[0], ))
        ret['masks'] = masks
        ret['mask_flag'] = mask_flag
        ret['dense_frame_flag'] = torch.ones_like(ret['image_flags'])
        ret['seg_image_num'] = torch.tensor([0])
        
        return ret
    
    if dataset_name == "ref_avs":
        masks, mask_flag = process_ref_avs(ret, data_item)
    else:
        raise NotImplementedError
    
    ret['masks'] = masks
    ret['mask_flag'] = mask_flag

    return ret


def process_ref_avs(ret, data_item):
    assert 'mask_path' in data_item, "Key 'mask_path' not in dict"
    if data_item['mask_path'] is None:
        masks = torch.zeros((ret['image_flags'].shape[0], MASK_SIZE, MASK_SIZE))
        mask_flag = torch.zeros((ret['image_flags'].shape[0], ))
    else:
        mask_path = data_item['mask_path']
        mask_file_list = sorted([f for f in os.listdir(mask_path) if f.endswith('.png')])

        transform = T.Compose([
            T.Resize((MASK_SIZE, MASK_SIZE), interpolation=InterpolationMode.NEAREST),
            T.ToTensor()
        ])

        masks = []
        for file_name in mask_file_list:
            file_path = os.path.join(mask_path, file_name)
            image = Image.open(file_path).convert('L')  # convert to gray
            
            mask = transform(image)
            masks.append(mask)

        masks = torch.cat(masks, dim=0)
        mask_flag = torch.ones((masks.shape[0], ))
    return masks, mask_flag


if __name__ == "__main__":
    data_item = {"id": 16, "image": ["-8ZG1rJYPXs_210000_220000/frames/0.jpg", "-8ZG1rJYPXs_210000_220000/frames/1.jpg", "-8ZG1rJYPXs_210000_220000/frames/2.jpg", "-8ZG1rJYPXs_210000_220000/frames/3.jpg", "-8ZG1rJYPXs_210000_220000/frames/4.jpg", "-8ZG1rJYPXs_210000_220000/frames/5.jpg", "-8ZG1rJYPXs_210000_220000/frames/6.jpg", "-8ZG1rJYPXs_210000_220000/frames/7.jpg", "-8ZG1rJYPXs_210000_220000/frames/8.jpg", "-8ZG1rJYPXs_210000_220000/frames/9.jpg"], "audio": "-8ZG1rJYPXs_210000_220000/audio.wav", "audio_length": [500, 250, 252], "conversations": [{"from": "human", "value": "Frame1: <image>\nFrame2: <image>\nFrame3: <image>\nFrame4: <image>\nFrame5: <image>\nFrame6: <image>\nFrame7: <image>\nFrame8: <image>\nFrame9: <image>\nFrame10: <image><audio>\nPlease segment the object this sentence describes in this video: <ref>The object making shorter sound duration than the cello.</ref>"}, {"from": "gpt", "value": "<ref>The object making shorter sound duration than the cello.</ref><seg>"}], "dataset_name": "ref_avs", "mask_path": "data/segmentation/refavs/gt_mask/-8ZG1rJYPXs_210000_220000/fid_1"}
    ret = dict()
    parse_mask_annotation(ret, data_item)
