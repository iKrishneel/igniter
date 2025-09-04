#!/usr/bin/env python

import os.path as osp
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from ..io.s3_client import S3Client
from ..logger import logger
from ..utils import check_str
from .coco import COCO

__all__ = ['S3Dataset', 'S3CocoDataset']


class S3Dataset(Dataset):
    def __init__(self, bucket_name: str, **kwargs):
        super(S3Dataset, self).__init__()
        self.client = S3Client(bucket_name)

    def load_image(self, filename: str) -> Type[Any]:
        check_str(filename, 'Filename is required')
        return self.client(filename)

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError('Not yet implemented')

    @abstractmethod
    def __len__(self):
        raise NotImplementedError('Not yet implemented')


class S3CocoDataset(S3Dataset):
    def __init__(
        self, bucket_name: str, root: str, anno_fn: str, transforms: Optional[Callable] = None, **kwargs
    ) -> None:
        check_str(anno_fn)
        assert anno_fn.split('.')[1] == 'json', f'Expects json file but got {anno_fn}'
        super(S3CocoDataset, self).__init__(bucket_name, **kwargs)

        self.root = root
        self.coco = COCO(self.client, anno_fn)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.apply_transforms = kwargs.get('apply_transforms', True)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        while True:
            try:
                iid = self.ids[index]
                image, target = self._load(iid)
                assert len(target) > 0
                break
            except Exception as e:
                logger.warning(f'{e} for iid: {iid}')
                index = np.random.choice(np.arange(len(self.ids)))
                
        if self.transforms is not None and self.apply_transforms:
            image, target = self.transforms(image, target)

        return image, target

    def _load(self, iid: int) -> Tuple[Any, ...]:
        file_name = osp.join(self.root, self.coco.loadImgs(iid)[0]['file_name'])
        image = self.load_image(file_name)
        image = Image.fromarray(image).convert('RGB') if not isinstance(image, Image.Image) else image  # type: ignore
        target = self.coco.loadAnns(self.coco.getAnnIds(iid))
        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class S3CocoDatasetV2(S3CocoDataset):

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        self.apply_transforms = False
        image, targets = super(S3CocoDatasetV2, self).__getitem__(index)

        image, targets = self._preprocess(image, targets)

        if self.transforms is not None:
            image, targets = self.transforms(image, targets)

        return image, targets

    def _preprocess(self, image: Image, targets: List[Dict[str, Any]]) -> Tuple[Any]:
        from torchvision.tv_tensors import BoundingBoxes, Mask

        im_hw = image.size[::-1] if isinstance(image, Image.Image) else image.shape[1:]
        bboxes, masks, category_names, category_ids = [], [], [], []

        for target in targets:
            # mask_rles = mask_utils.frPyObjects(target['segmentation'], *im_hw)
            try:
                mask = decode_coco_mask(target['segmentation'], *im_hw)
            except TypeError:
                breakpoint()
            
            category_name = self.coco.cats[target['category_id']]['name']
            
            bboxes.append(target['bbox'])
            masks.append(mask)
            category_names.append(category_name)
            category_ids.append(target['category_id'])

        bboxes = BoundingBoxes(bboxes, format='XYWH', canvas_size=im_hw)
        masks = Mask(masks, requires_grad=False)

        annotations = {
            'bboxes': bboxes,
            'masks': masks,
            'category_names': category_names,
            'category_ids': [target['category_id'] for target in targets],
            'ids': [target['id'] for target in targets]
        }
        return image, annotations

    
def decode_coco_mask(segmentation: Union[List, Dict], height: int = None, width: int = None):
    from pycocotools import mask as mask_utils    

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
    elif isinstance(segmentation, dict):
        if isinstance(segmentation['counts'], list):
            rle = mask_utils.frPyObjects(segmentation, segmentation['size'][0], segmentation['size'][1])
            mask = mask_utils.decode(rle)
        else:
            mask = mask_utils.decode(segmentation)
    else:
        raise TypeError(f'Unknown segmentation format: {type(segmentation)}')
    
    return mask.astype(np.uint8)
