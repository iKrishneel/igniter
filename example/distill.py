#!/usr/bin/env python

import os
from typing import Any, Dict, List, Optional
from io import BytesIO
import time

import torch
import torch.nn as nn
from einops import rearrange

import numpy as np

import hydra
from timm.models import swin_tiny_patch4_window7_224

from igniter import initiate
from igniter.builder import trainer
from igniter.datasets import S3CocoDataset
from igniter.registry import model_registry, proc_registry, dataset_registry


@model_registry('swin')
class SwinTP4W7(nn.Module):
    def __init__(self, in_size: List[int] = [896, 896], **kwargs: Optional[Dict[str, Any]]):
        super(SwinTP4W7, self).__init__()

        model = swin_tiny_patch4_window7_224(pretrained=False, img_size=in_size)
        for attr in ['head']:
            delattr(model, attr)

        self.model = model
        self.in_size = list(in_size)

        target_size = kwargs.get('target_size', [64, 64])
        self.stride = kwargs.get('stride', 32)
        self.target_size = (
            [target_size, target_size] if not isinstance(target_size, list) or len(target_size) == 1 else target_size
        )

        self.conv = nn.Conv2d(768, 256, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        _, _, h, w = x.shape
        if [h, w] != self.in_size:
            x = nn.functional.interpolate(x, self.in_size, mode='nearest')

        x = self.model.patch_embed(x)
        x = self.model.layers(x)
        x = self.model.norm(x)

        h1, w1 = self.in_size[1] // self.stride, self.in_size[0] // self.stride
        x = rearrange(x, 'b (h1 w1) c -> b c h1 w1', h1=h1, w1=w1)
        x = nn.functional.interpolate(x, self.target_size, mode='bilinear')
        x = self.conv(x)

        if self.training or target is not None:
            assert target is not None
            return self.losses(x, target)

        return x

    def losses(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # kl_loss = nn.KLDivLoss(reduction='batchmean')
        # x = nn.functional.log_softmax(x, dim=1)
        # loss = kl_loss(x, nn.functional.softmax(target, dim=1))

        loss = nn.functional.l1_loss(x, target)
        return {'loss': loss}


@dataset_registry('coco')
class S3CocoDatasetSam(S3CocoDataset):
    def __init__(self, *args, **kwargs):
        super(S3CocoDatasetSam, self).__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        iid = self.ids[index]

        while True:
            time.sleep(0.1)
            try:
                image, target = self._load(iid)
            except Exception as e:
                logger.warning(f'{e} for iid: {iid}')
                iid = np.random.choice(iid)
        # temp
        filename = f'perception/sam/coco/{self.root.split(os.sep)[-1]}/features/'
        filename = filename + f'{str(iid).zfill(12)}.pt'

        contents = self.client.get(filename, False)
        buffer = BytesIO(contents)
        sam_feats = torch.load(buffer, map_location=torch.device('cpu'))
        return {'image': image, 'sam_feats': sam_feats, 'filename': filename}


@proc_registry('collate_data')
def collate_data(batches) -> List[torch.Tensor]:
    images = torch.stack([batch['image'] for batch in batches])
    targets = torch.stack([batch['sam_feats'] for batch in batches])
    return images, targets


@proc_registry('accuracy')
def metric(engine, name):
    from ignite.metrics import Accuracy

    def _output_transform(data):
        import IPython

        IPython.embed()

    Accuracy(output_transform=_output_transform).attach(engine, name)


initiate('./configs/swin.yaml')
