#!/usr/bin/env python

from io import BytesIO
from typing import Any, Dict

import numpy as np

from igniter.registry import Registry

s3_utils_registry = Registry()


lut = {
    'image': 'decode_pil_image',
    'image/jpeg': 'decode_pil_image',
    'json': 'decode_json',
    'application/json': 'decode_json',
    'torch': 'decode_torch_weights',
}


@s3_utils_registry
def decode_cv_image(content) -> np.ndarray:
    import cv2 as cv

    data = np.frombuffer(content, np.uint8)
    return cv.imdecode(data, cv.IMREAD_ANYCOLOR)


@s3_utils_registry
def decode_pil_image(content) -> Any:
    from PIL import Image

    return Image.open(BytesIO(content))


@s3_utils_registry
def decode_json(content) -> Dict[str, Any]:
    import json

    return json.loads(content)


@s3_utils_registry
def decode_torch_weights(content) -> Dict[str, Any]:
    import torch

    buffer = BytesIO(content)
    return torch.load(buffer, map_location=torch.device('cpu'))
