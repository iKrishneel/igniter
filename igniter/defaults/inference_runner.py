#!/usr/bin/env python

import os
import os.path as osp
import time
from dataclasses import InitVar, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from igniter.engine import InferenceEngine
from igniter.logger import logger
from igniter.registry import func_registry

IMAGE_EXTS: List[str] = ['.jpg', '.png', '.jpeg']
VIDEO_EXTS: List[str] = ['.avi', '.mp4', '.mov']
INPUT_FMTS: List[str] = ['RGB', 'BGR', 'GRAY', 'MONO']


def build_hook(name: str, **kwargs: Dict[str, Any]) -> Callable:
    return partial(func_registry[name], **kwargs)


@dataclass
class Inference(object):
    filename: InitVar[str]
    engine: InferenceEngine
    threshold: Optional[float] = 0.0
    input_fmt: Optional[str] = 'RGB'
    save: Optional[bool] = False
    save_dir: Optional[str] = None
    _pre_hooks: List[Callable] = field(default_factory=lambda: [])
    _post_hooks: List[Callable] = field(default_factory=lambda: [])

    def __post_init__(self, filename: str) -> None:
        # assert osp.isfile(filename) or osp.isdir(filename), f'Invalid path: {filename}!'
        if osp.isfile(filename):
            supported_exts, ext = IMAGE_EXTS + VIDEO_EXTS, osp.splitext(filename)[1]
            assert ext.lower() in supported_exts, f'Invalid file {filename}. Supported file types are {supported_exts}'
            assert self.input_fmt.upper() in ['RGB', 'BGR', 'GRAY', 'MONO'], f'Invalid input format {self.input_fmt}'
            self._loader = partial(self.load_image if ext in IMAGE_EXTS else self.load_video, filename=filename)
            # self._ext = ext
        elif osp.isdir(filename):
            filenames = [f for f in Path(filename).iterdir() if f.suffix.lower() in IMAGE_EXTS]
            self._loader = partial(self.load_images, filenames=filenames)
        else:
            raise TypeError(f'Invalid path: {filename}')

        if self.save:
            assert osp.isdir(self.save_dir), f'{self.save_dir} is not a directory'
            os.makedirs(self.save_dir, exist_ok=True)

        assert self.engine is not None

    def __call__(self) -> None:
        self._loader()

    def run(self) -> None:
        self()

    def load_image(self, filename: str) -> None:
        image = Image.open(filename).convert(self.input_fmt)
        self.process(image, filename=filename)

    def load_video(self, filename: str) -> None:
        logger.info(f'Loading video from: {filename}')
        start_time = time.time()
        cap = cv.VideoCapture(filename)
        counter = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.process(frame, str(counter))
            counter += 1
        cap.release()
        logger.info(f'Total Processing time: {time.time() - start_time}')
        logger.info('Completed!')

    def load_images(self, filenames: List[str]):
        for filename in filenames:
            self.load_image(filename)

    def process(self, image: Union[Image.Image, np.ndarray], filename: str = None) -> Any:
        image = np.asarray(image) if not isinstance(image, np.ndarray) else image
        start_time = time.time()
        pred = self._process(image, filename)
        logger.info(f'Inference time: {time.time() - start_time}')
        return pred

    @torch.inference_mode()
    def _process(self, image: Image, filename: str = None) -> Any:
        image = self._run_hooks(image, self._pre_hooks)
        pred = self.engine(image)
        # TODO(iKrishneel): make this into a collate function
        data = {'image': image, 'pred': pred, 'filename': filename}
        pred = self._run_hooks(data, self._post_hooks)
        return pred

    def register_forward_pre_hook(self, func: Union[Callable, str]) -> None:
        assert callable(func)
        self._pre_hooks.append(func)

    def register_forward_post_hook(self, func: Union[Callable, str]) -> None:
        assert callable(func)
        self._post_hooks.append(func)

    @staticmethod
    def _run_hooks(data: Dict[str, Any], hooks) -> Any:
        for hook in hooks:
            _ = hook(data)
            data = _ or data
        return data
