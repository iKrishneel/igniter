#!/usr/bin/env python

from typing import Any, Dict, List, Type
from dataclasses import dataclass, field

import threading
from io import BytesIO
import concurrent.futures as cf

import numpy as np
import cv2 as cv
import json

import boto3
from igniter.logger import logger


S3 = Type['boto3.client.S3']


@dataclass
class S3Client(object):
    bucket_name: str
    _s3: S3 = None

    def __post_init__(self) -> None:
        assert len(self.bucket_name) > 0, f'Invalid bucket name'
        logger.info(f'Data source is s3://{self.bucket_name}')
        self._s3 = boto3.client('s3')

    def get(self, filename: str, ret_raw: bool = True):
        s3_file = self.client.get_object(Bucket=self.bucket_name, Key=filename)
        if ret_raw:
            return s3_file
        return self._read(s3_file)

    def __call__(self, filename: str) -> Type[Any]:
        return self.load_file(filename)

    def __getitem__(self, filename: str) -> Type[Any]:
        return self.load_file(filename)

    def load_file(self, filename: str):
        assert len(filename) > 0, f'Invalid filename'
        return self.decode_file(self.get(filename))

    def decode_file(self, s3_file) -> Type[Any]:
        content_type = s3_file['ResponseMetadata']['HTTPHeaders']['content-type']
        if 'image' in content_type:
            val = self.decode_image(s3_file)
        elif 'json' in content_type:
            val = self.decode_json(s3_file)
        else:
            raise TypeError(f'Unknown file type {content_type}')
        return val

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

    def decode_image(self, s3_file) -> np.ndarray:
        content = self._read(s3_file)
        data = np.frombuffer(content, np.uint8)
        return cv.imdecode(data, cv.IMREAD_COLOR)

    def decode_json(self, s3_file) -> Dict[str, Any]:
        content = self._read(s3_file)
        return json.loads(content)

    @property
    def client(self) -> S3:
        return self._s3


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
