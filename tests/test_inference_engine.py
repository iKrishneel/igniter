#!/usr/bin/env python

import pytest
import os
import os.path as osp
import shutil
import functools

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf

from igniter.defaults import InferenceEngine
from igniter.registry import model_registry

ROOT: str = '/tmp/igniter/test_inference_engine/'
WPATH: str = osp.join(ROOT, 'test_model.pth')
os.makedirs(ROOT, exist_ok=True)


@model_registry('test_model')
class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        return torch.sigmoid(x)


def setup_test():
    model = ExampleModel()
    torch.save({'model': model.state_dict()}, WPATH)

    config = {
        'device': 'cpu',
        'models': {
            'test_model': None,
        },
        'build': {'model': 'test_model', 'test_model': {'inference': {'weights': WPATH}}},
    }

    config_file = osp.join(ROOT, 'config.yaml')
    config = OmegaConf.create(config)
    OmegaConf.save(config, config_file)

    return {'config_file': config_file, 'model': model}


@pytest.fixture(scope='session', autouse=True)
def config(request):
    def clean_up():
        shutil.rmtree(ROOT)

    request.addfinalizer(clean_up)

    return setup_test()


def assert_all(func):
    @functools.wraps(func)
    def wrapper(config):
        ie = func(config)

        model = config['model']
        assert isinstance(ie.model, ExampleModel)
        assert len(ie.model.state_dict()) == len(model.state_dict())

        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], ie.model.state_dict()[key])

    return wrapper


@assert_all
def test_with_config(config):
    return InferenceEngine(config_file=config['config_file'])


@assert_all
def test_with_config_and_weights(config):
    return InferenceEngine(config_file=config['config_file'], weights=WPATH)


@assert_all
def test_with_logdir(config):
    return InferenceEngine(log_dir=ROOT, extension='.pth')


def test_with_no_input(config):
    with pytest.raises(AssertionError) as e:
        ie = InferenceEngine()
    assert str(e.value) == 'Must provide either the log_dir or the config file'


def test_with_invalid_logdir(config):
    with pytest.raises(IndexError) as e:
        ie = InferenceEngine(log_dir='/tmp/')
    assert str(e.value) == 'list index out of range'


def test_with_invalid_config(config):
    with pytest.raises(TypeError) as e:
        ie = InferenceEngine(config_file='/tmp/config.yaml')
    assert str(e.value) == 'Invalid config_file /tmp/config.yaml'

    with pytest.raises(TypeError) as e:
        ie = InferenceEngine(config_file='/tmp/')
    assert str(e.value) == 'Invalid config_file /tmp/'


def test_with_input(config):
    image = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)

    ie = InferenceEngine(log_dir=ROOT, extension='.pth')
    y = ie(image)

    assert len(y.shape) == 3

    assert y.shape[0] == 1
    assert y.shape[1] == 224
    assert y.shape[2] == 224
