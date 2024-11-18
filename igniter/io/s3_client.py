#!/usr/bin/env python

import os
import re
import threading
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Optional, Type, Union

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

from igniter.logger import logger

from .s3_utils import lut, s3_utils_registry


@dataclass
class S3Client(object):
    bucket_name: str

    def __post_init__(self) -> None:
        assert len(self.bucket_name) > 0, f'Invalid bucket name {self.bucket_name}'
        logger.info(f'Data source is s3://{self.bucket_name}')

    def get(self, filename: str, ret_raw: bool = True):
        s3_file = self.client.get_object(Bucket=self.bucket_name, Key=filename)
        if ret_raw:
            return s3_file
        return self._read(s3_file)

    def __call__(self, filename: str, decoder: Optional[Union[Callable, str]] = None) -> Type[Any]:
        return self.load_file(filename, decoder)

    def __getitem__(self, filename: str) -> Type[Any]:
        return self(filename)

    def __reduce__(self):
        return (self.__class__, (self.bucket_name,))

    def download(self, file_key: str, filename: str):
        """
        Downloads a file from an S3 bucket and saves it locally, tracking the progress.

        :param file_key: The key of the file to download within the bucket.
        :param filename: The local path where the file will be saved.
        """
        if os.path.isdir(filename):
            filename = os.path.join(filename, file_key.split('/')[-1])

        def last_n(text: str, n: int = 24):
            n = max(1, n)
            return '...' + text[-n:] if len(text) > n else text

        try:
            object_info = self.client.get_object(Bucket=self.bucket_name, Key=file_key)
            total_bytes = object_info['ContentLength']

            with tqdm(
                total=total_bytes, unit='B', unit_scale=True, unit_divisor=1024, desc=f'Downloading {last_n(file_key)}'
            ) as progress_bar:

                def progress_callback(bytes_transferred):
                    progress_bar.update(bytes_transferred)

                self.client.download_file(self.bucket_name, file_key, filename, Callback=progress_callback)
        except KeyboardInterrupt:
            logger.info('Download interrupted. Cleaning up ...')
            if os.path.isfile(filename):
                os.remove(filename)
        except Exception as e:
            logger.error(f'Error downloading file "{file_key}": {e}')

    def load_file(self, filename: str, decoder: Optional[Union[Callable, str]] = None):
        assert len(filename) > 0, f'Invalid filename {filename}'
        try:
            return self.decode_file(self.get(filename), decoder)
        except ClientError as e:
            print(f'{e}\nFile Not Found! {filename}')
            return {}

    def upload(self, filename: str, object_key: str) -> None:
        assert os.path.isfile(filename), f'{filename} not found!'
        assert isinstance(object_key, str)

        def _is_valid_filename(string: str) -> bool:
            pattern = r"^(?:[\w,\s-]+/)*[\w,\s-]+\.[A-Za-z]{2,5}$"
            return bool(re.match(pattern, string))

        object_key = object_key.lstrip('/')
        object_key = (
            os.path.join(object_key, os.path.basename(filename)) if not _is_valid_filename(object_key) else object_key
        )

        logger.info(f'Uploading {filename} to s3://{self.bucket_name}/{object_key}')
        self.client.upload_file(filename, Bucket=self.bucket_name, Key=object_key)
        logger.info('File Uploaded!')

    def decode_file(self, s3_file, decoder: Optional[Union[Callable[..., Any], str]] = None) -> Type[Any]:
        content_type = s3_file['ResponseMetadata']['HTTPHeaders']['content-type']
        content = self._read(s3_file)

        decoder = lut.get(content_type, None) if not decoder else decoder
        assert decoder, f'Decoder for content type {content_type} is unknown'
        func = s3_utils_registry[decoder]  # type: ignore

        assert func, 'Unknown decoder function'
        return func(content)

    def write(self, buffer: BytesIO, path: str, same_thread: bool = True) -> None:
        if same_thread:
            self._write(buffer, path)
        else:
            thread = threading.Thread(target=self._write, args=(buffer, path))
            thread.start()

    def _write(self, buffer: BytesIO, path: str):
        assert isinstance(buffer, BytesIO), f'Except type {type(BytesIO)} but got {type(buffer)}'
        assert len(path), 'Invalid path: {path}'
        response = self.client.put_object(Bucket=self.bucket_name, Key=path, Body=buffer.getvalue())
        return response

    def _read(self, s3_file):
        return s3_file['Body'].read()

    def ls(self, directory: str):
        return self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=directory)

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

    import torch
    from torchvision.models import resnet18

    buffer = BytesIO()

    m = resnet18()
    torch.save(m.state_dict(), buffer)

    import IPython

    IPython.embed()
