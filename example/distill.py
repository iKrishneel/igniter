#!/usr/bin/env python

import os
from typing import Any, Dict, List, Optional
from io import BytesIO

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
        self.in_size = in_size

        target_size = kwargs.get('target_size', [64, 64])
        self.stride = kwargs.get('stride', 32)
        self.target_size = (
            [target_size, target_size] if not isinstance(target_size, list) or len(target_size) == 1 else target_size
        )

        self.conv = nn.Conv2d(768, 256, kernel_size=3, padding=1)

    @classmethod
    def build(cls, cfg):
        name = cfg.build.model
        attrs = dict(in_size=cfg.models[name].in_size)
        return cls(**attrs)

    def forward(self, x, target: Dict[str, Any] = None):
        _, _, h, w = x.shape
        x = nn.functional.interpolate(x, self.in_size, mode='nearest')

        x = self.model.patch_embed(x)
        x = self.model.layers(x)
        x = self.model.norm(x)

        h1, w1 = self.in_size[1] // self.stride, self.in_size[0] // self.stride
        x = rearrange(x, 'b (h1 w1) c -> b c h1 w1', h1=h1, w1=w1)

        x = nn.functional.interpolate(x, self.target_size, mode='bilinear')
        x = self.conv(x)

        if self.training:
            assert target is not None
            return self.losses(x, target)

        return x

    def losses(self, x: torch.Tensor, target: Dict[str, None]):
        k = target['sam_feats']
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        x = nn.functional.log_softmax(x, dim=1)
        k = nn.functional.softmax(k, dim=1)
        return kl_loss(x, k)


@dataset_registry('coco')
class S3CocoDatasetSam(S3CocoDataset):
    def __init__(self, *args, **kwargs):
        super(S3CocoDatasetSam, self).__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        iid = self.ids[index]
        image, target = super().__getitem__(index)

        # temp
        filename = f'perception/sam/coco/{self.root.split(os.sep)[-1]}/features/'
        filename = filename + f'{str(iid).zfill(12)}.pt'

        contents = self.client.get(filename, False)
        buffer = BytesIO(contents)
        sam_feats = torch.load(buffer)
        return {'image': image, 'sam_feats': sam_feats}


@proc_registry('update_model')
def update_model(engine, batch):
    engine._model.train()
    image, target = batch

    import IPython, sys

    IPython.embed()
    sys.exit()

    engine._optimizer.zero_grad()

    losses = engine._model(image, target)

    losses.backward()
    engine._optimizer.step()
    return losses.item()


initiate('./configs/swin.yaml')
