#!/usr/bin/env python

import os
import os.path as osp
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Union

import torch
from omegaconf import DictConfig

from .. import utils
from ..registry import io_registry
from .s3_client import S3Client


@io_registry('s3_writer')
class S3IO(S3Client):
    def __init__(self, bucket_name, root, **kwargs):
        assert len(root) > 0, 'Invalid root'
        self.cfg = kwargs.pop('cfg', None)
        super(S3IO, self).__init__(bucket_name, **kwargs)
        root = osp.normpath(root)
        self.root = root[1:] if root[0] == '/' else root

        extension = kwargs.pop('extension', 'pt')
        self.extension = '.' + extension if '.' not in extension else extension

    @classmethod
    def build(cls, io_cfg: DictConfig, cfg: DictConfig):
        return cls(bucket_name=io_cfg.bucket_name, root=io_cfg.root, cfg=cfg)

    def __call__(self, data: Any, filename: str):  # type: ignore
        assert len(filename) > 0, 'Invalid filename'

        if len(filename.split('.')) == 1:
            filename += self.extension

        # TODO: Check the data type and use appropriate writer
        self._save_tensors(data, filename)

    def _save_tensors(self, data: Union[Dict[str, Any], torch.Tensor], filename: str):
        buffer = BytesIO()
        torch.save(data, buffer)
        path = osp.join(self.root, filename)
        self.write(buffer, path, False)


@io_registry('file_writer')
@dataclass
class FileIO(object):
    root: str
    extension: str = '.pt'
    cfg: DictConfig = None

    def __post_init__(self):
        assert len(self.root) > 0, f'Directory {self.root} is not valid!'

    @classmethod
    def build(cls, io_cfg: DictConfig, cfg: DictConfig):
        return cls(root=io_cfg.root, cfg=cfg)

    def __call__(self, data: Any, filename: str):
        if utils.is_main_process():
            assert len(filename) > 0, 'Invalid filename'

            filename = osp.join(self.root, filename)
            os.makedirs(osp.dirname(filename), exist_ok=True)

            if self.extension not in filename:
                filename += self.extension

            # TODO: Check the data type and use appropriate writer
            self._save_tensors(data, filename)

        # torch.distributed.barrier()

    def _save_tensors(self, data: Union[Dict[str, Any], torch.Tensor], filename: str):
        torch.save(data, filename)
