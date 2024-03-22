#!/usr/bin/env python

import os.path as osp
from abc import abstractmethod
from typing import Any, Callable, Optional, Tuple, Type

import numpy as np
from PIL import Image
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
        self.apply_transforms = True

        # TODO: Add target transforms
        if transforms:
            import warnings

            warnings.warn('Target transforms is not yet implemented')

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        while True:
            try:
                iid = self.ids[index]
                image, target = self._load(iid)
                if self.transforms and self.apply_transforms:
                    image = self.transforms(image)
                return image, target
            except Exception as e:
                logger.warning(f'{e} for iid: {iid}')
                index = np.random.choice(np.arange(len(self.ids)))

    def _load(self, iid: int) -> Tuple[Any, ...]:
        file_name = osp.join(self.root, self.coco.loadImgs(iid)[0]['file_name'])
        image = self.load_image(file_name)
        image = Image.fromarray(image).convert('RGB') if not isinstance(image, Image.Image) else image  # type: ignore
        target = self.coco.loadAnns(self.coco.getAnnIds(iid))
        return image, target

    def __len__(self) -> int:
        return len(self.ids)
