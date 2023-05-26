#!/usr/bin/env python

import os
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO
import time

import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from einops import rearrange

import numpy as np

from timm.models import swin_tiny_patch4_window7_224

from segment_anything.modeling.common import LayerNorm2d

from igniter import initiate
from igniter.datasets import S3CocoDataset
from igniter.registry import model_registry, func_registry, dataset_registry, transform_registry


@model_registry('swin')
class SwinTP4W7(nn.Module):
    def __init__(self, in_size: List[int] = [896, 896], is_resize: bool = False, **kwargs: Optional[Dict[str, Any]]):
        assert in_size[0] == in_size[1], f'Current Implementation only supports square image {in_size}'
        super(SwinTP4W7, self).__init__()
        model = swin_tiny_patch4_window7_224(pretrained=False, img_size=in_size)
        for attr in ['head']:
            delattr(model, attr)

        self.model = model
        self.in_size = list(in_size)
        self.is_resize = is_resize

        out_channels = kwargs.get('out_channels', 256)
        target_size = kwargs.get('target_size', [64, 64])
        self.stride = kwargs.get('stride', 32)
        self.target_size = (
            [target_size, target_size] if not isinstance(target_size, list) or len(target_size) == 1 else target_size
        )

        self.neck = nn.Sequential(
            nn.Conv2d(768, out_channels, kernel_size=3, padding=1, bias=False), LayerNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        x = self.model.patch_embed(x)
        x = self.model.layers(x)
        x = self.model.norm(x)

        h1, w1 = self.in_size[1] // self.stride, self.in_size[0] // self.stride
        x = rearrange(x, 'b (h1 w1) c -> b c h1 w1', h1=h1, w1=w1)
        x = nn.functional.interpolate(x, self.target_size, mode='bilinear')
        x = self.neck(x)

        if target is not None:
            losses = self.losses(x, target)
            if self.training:
                return losses
            return x, self.losses(x, target)

        return x

    def losses(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        x = nn.functional.log_softmax(x, dim=1)
        loss = kl_loss(x, nn.functional.softmax(target, dim=1))
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
                break
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


@transform_registry
class ResizeLongestSide(nn.Module):
    def __init__(self, size: int) -> None:
        super(ResizeLongestSide, self).__init__()
        self.size = size

    def forward(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image) if not isinstance(image, np.ndarray) else image
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return np.array(TF.resize(TF.to_pil_image(image), target_size))

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


@transform_registry
class PadToSize(nn.Module):
    def __init__(self, size: int):
        super(PadToSize, self).__init__()
        self.size = size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        padh, padw = self.size - h, self.size - w
        return nn.functional.pad(image, (0, padw, 0, padh))


@func_registry('collate_data')
def collate_data(batches) -> List[torch.Tensor]:
    images = torch.stack([batch['image'] for batch in batches])
    targets = torch.stack([batch['sam_feats'] for batch in batches])
    return images, targets


@func_registry('accuracy')
def metric(engine, name):
    from ignite.metrics import Accuracy

    def _output_transform(data):
        import IPython

        IPython.embed()

    Accuracy(output_transform=_output_transform).attach(engine, name)


initiate('./configs/swin.yaml')
