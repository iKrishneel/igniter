#!/usr/bin/env python

from typing import Any, Dict, Union
import os.path as osp
import torch
from io import BytesIO

from .s3_client import S3Client
from ..registry import io_registry


@io_registry('s3_writer')
class S3IO(S3Client):
    def __init__(self, bucket_name, root, **kwargs):
        assert len(root) > 0, 'Invalid root'
        super(S3IO, self).__init__(bucket_name, **kwargs)
        root = osp.normpath(root)
        self.root = root[1:] if root[0] == '/' else root

        extension = kwargs.pop('extension', 'pt')
        self.extension = '.' + extension if '.' not in extension else extension

    @classmethod
    def build(cls, io_cfg):
        return cls(bucket_name=io_cfg.bucket_name, root=io_cfg.root)

    def __call__(self, data: Any, filename: str):
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
