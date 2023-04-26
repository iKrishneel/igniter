#!/usr/bin/env python

from typing import Tuple
import os.path as osp
import numpy as np
from torch.utils.data import Dataset

from PIL import Image

from .s3_client import S3Client
from .coco import COCO
from ..utils import check_str


__all__ = ['S3Dataset', 'S3CocoDataset']


class S3Dataset(Dataset):
    def __init__(self, bucket_name: str, **kwargs):
        super(S3Dataset, self).__init__()
        self.client = S3Client(bucket_name)

    def load_image(self, filename: str) -> np.ndarray:
        check_str(filename, f'Filename is required')
        return self.client(filename)

    def __getitem__(self, index: int):
        raise NotImplementedError('Not yet implemented')


class S3CocoDataset(S3Dataset):
    def __init__(self, bucket_name: str, root: str, anno_fn: str, **kwargs) -> None:
        check_str(anno_fn)
        super(S3CocoDataset, self).__init__(bucket_name, **kwargs)

        self.root = root
        self.coco = COCO(self.client, anno_fn)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index: int) -> Tuple:
        iid = self.ids[index]
        file_name = osp.join(self.root, self.coco.loadImgs(iid)[0]['file_name'])
        image = self.load_image(file_name)
        image = Image.fromarray(image).convert('RGB')
        target = self.coco.loadAnns(self.coco.getAnnIds(iid))
        return image, target

    def __len__(self) -> int:
        return len(self.ids)
