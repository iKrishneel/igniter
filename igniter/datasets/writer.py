#!/usr/bin/env python

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
    def build(cls, cfg):
        args = cfg.io.args
        return cls(bucket_name=args.bucket_name, root=args.root)

    def __call__(self, tensor: torch.Tensor, filename: str):
        assert len(filename) > 0, 'Invalid filename'

        if len(filename.split('.')) == 1:
            filename += self.extension

        buffer = BytesIO()
        torch.save(tensor, buffer)
        path = osp.join(self.root, filename)
        self.write(buffer, path, False)
