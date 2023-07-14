#!/usr/bin/env python

import pytest

from io import BytesIO
import base64
import numpy as np
from PIL import Image
import cv2 as cv

from igniter.io import S3Client


def to_s3_response(body):
    return {'ResponseMetadata': {'HTTPHeaders': {'content-type': 'image/jpeg'}}, 'Body': body}


def mock_read(self, data):
    return data['Body']


@pytest.fixture
def image():
    return np.random.randint(0, 255, (480, 640, 3)).astype(np.uint8)


def test_s3_client_cv_image(image, mocker):
    def mock_get(self, filename: str, ret_raw: bool = True):
        image_bytes = np.array(cv.imencode('.jpg', image)[1]).tobytes()
        return to_s3_response(image_bytes)

    m = mocker.patch('igniter.io.s3_client.S3Client.get', mock_get)
    m = mocker.patch('igniter.io.s3_client.S3Client._read', mock_read)

    client = S3Client('mybucket')
    response = client('folder', decoder='decode_cv_image')

    assert image.shape == response.shape


def test_s3_client_pil_image(image, mocker):
    image = Image.fromarray(image)

    def mock_get(self, filename: str, ret_raw: bool = True):
        byte_stream = BytesIO()
        image.save(byte_stream, format='JPEG')
        image_bytes = byte_stream.getvalue()
        byte_stream.close()
        return to_s3_response(image_bytes)

    m = mocker.patch('igniter.io.s3_client.S3Client.get', mock_get)
    m = mocker.patch('igniter.io.s3_client.S3Client._read', mock_read)

    client = S3Client('mybucket')
    response = client('folder')
    assert image.size == response.size

    response = client('folder', decoder='decode_pil_image')
    assert image.size == response.size


def test_s3_client_json(mocker):
    data = {
        'annotations': [
            {'filename': 'image1.jpg', 'height:': 480, 'width': 640},
            {'filename': 'image2.jpg', 'height:': 774, 'width': 840},
        ]
    }

    def mock_get(self, filename: str, ret_raw: bool = True):
        import json

        byte_array = json.dumps(data).encode()
        return to_s3_response(byte_array)

    m = mocker.patch('igniter.io.s3_client.S3Client.get', mock_get)
    m = mocker.patch('igniter.io.s3_client.S3Client._read', mock_read)

    client = S3Client('mybucket')
    response = client('folder', decoder='decode_json')
    assert data == response


def test_s3_client_torch_weights(mocker):
    import torch

    state_dict = torch.nn.Conv2d(3, 64, 3).state_dict()

    def mock_get(self, filename: str, ret_raw: bool = True):
        byte_stream = BytesIO()
        torch.save(state_dict, byte_stream)
        byte_state = byte_stream.getvalue()
        byte_stream.close()
        return to_s3_response(byte_state)

    m = mocker.patch('igniter.io.s3_client.S3Client.get', mock_get)
    m = mocker.patch('igniter.io.s3_client.S3Client._read', mock_read)

    client = S3Client('mybucket')
    response = client('folder', decoder='decode_torch_weights')

    for key in state_dict:
        assert torch.all(state_dict[key] == response[key])
