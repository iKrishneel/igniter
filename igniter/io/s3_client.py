#!/usr/bin/env python

from typing import Any, Dict, List, Type, Callable, Optional
from dataclasses import dataclass, field

import threading
from io import BytesIO
import concurrent.futures as cf

import boto3
from igniter.logger import logger

from .s3_utils import s3_utils_registry


@dataclass
class S3Client(object):
    bucket_name: str
    decoder_func: Optional[Callable[[str], None]] = None

    def __post_init__(self) -> None:
        assert len(self.bucket_name) > 0, f'Invalid bucket name'
        logger.info(f'Data source is s3://{self.bucket_name}')

        if self.decoder_func and isinstance(self.decoder_func, str):
            self.decoder_func = s3_utils_registry[self.decoder_func]

    def get(self, filename: str, ret_raw: bool = True):
        s3_file = self.client.get_object(Bucket=self.bucket_name, Key=filename)
        if ret_raw:
            return s3_file
        return self._read(s3_file)

    def __call__(self, filename: str) -> Type[Any]:
        return self.load_file(filename)

    def __getitem__(self, filename: str) -> Type[Any]:
        return self(filename)

    def __reduce__(self):
        return (self.__class__, (self.bucket_name,))

    def load_file(self, filename: str):
        assert len(filename) > 0, f'Invalid filename'
        return self.decode_file(self.get(filename))

    def decode_file(self, s3_file) -> Type[Any]:
        content_type = s3_file['ResponseMetadata']['HTTPHeaders']['content-type']
        content = self._read(s3_file)

        func = self.decoder_func
        if not self.decoder_func:
            # TODO: Remove hardcoded conditions
            if 'image' in content_type:
                func_name = 'decode_cv_image'
            elif 'json' in content_type:
                func_name = 'decode_json'
            else:
                raise TypeError(f'Unknown file type {content_type}')
            func = s3_utils_registry[func_name]
        assert func, 'Unknown decoder function'
        return func(content)

    def write(self, buffer: BytesIO, path: str, same_thread: bool = True) -> None:
        if same_thread:
            self._write(buffer, path)
        else:
            thread = threading.Thread(target=self._write, args=(buffer, path))
            thread.start()

    def _write(self, buffer: BytesIO, path: str):
        assert isinstance(buffer, BytesIO), f'Except type {type(ByteIO)} but got {type(buffer)}'
        assert len(path), 'Invalid path: {path}'
        response = self.client.put_object(Bucket=self.bucket_name, Key=path, Body=buffer.getvalue())
        return response

    def _read(self, s3_file):
        return s3_file['Body'].read()

    @property
    def client(self):
        return boto3.client('s3')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--root', type=str, default='perception/datasets/coco/')
    args = parser.parse_args()

    s3 = S3Client(bucket_name=args.bucket)

    # im = s3[args.root + 'train2017/000000005180.jpg']
    # js = s3['instances_val2017.json']

    from torchvision.models import resnet18
    import torch

    buffer = BytesIO()

    m = resnet18()
    torch.save(m.state_dict(), buffer)

    import IPython

    IPython.embed()
