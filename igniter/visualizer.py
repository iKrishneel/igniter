#!/usr/bin/env python

from typing import List, Optional

import cv2 as cv
import numpy as np


def resize(data: List[np.ndarray], size: Optional[List[int]] = None) -> np.ndarray:
    if size is None:
        size = [0, 0]
        for d in data:
            size[0] = d.shape[0] if d.shape[0] > size[0] else size[0]
            size[1] = d.shape[1] if d.shape[1] > size[1] else size[1]
    return np.array([cv.resize(d, size[::-1]) for d in data])


def make_square_grid(in_data: List[np.ndarray], values: Optional[int] = None) -> np.ndarray:
    values = np.min(in_data) if not values else values
    data = resize(in_data)
    # data = np.array(data) if isinstance(data, list) else data
    n = int(np.ceil(np.sqrt(data.shape[0])))
    m = int(np.ceil(data.shape[0] / n))
    padding = ((0, n * m - data.shape[0]), (0, 1), (0, 1)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=values)
    data = data.reshape((m, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    return data.reshape((m * data.shape[1], n * data.shape[3]) + data.shape[4:])
