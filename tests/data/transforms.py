#!/usr/bin/env python

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as TF

from igniter.registry import transform_registry


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
