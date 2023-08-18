#!/usr/bin/env python

import functools

import numpy as np
import pytest
import torch
from conftest import ROOT
from data.model import ExampleModel

from igniter.engine import InferenceEngine


def assert_all(func):
    @functools.wraps(func)
    def wrapper(model, config_file):
        ie = func(model, config_file)

        assert isinstance(ie.model, ExampleModel)
        assert len(ie.model.state_dict()) == len(model.state_dict())

        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], ie.model.state_dict()[key])

    return wrapper


@assert_all
def test_with_config(model, config_file):
    return InferenceEngine(config_file=config_file)


@assert_all
def test_with_config_and_weights(model, config_file):
    wpth = '/tmp/igniter/tests/model.pth'
    return InferenceEngine(config_file=config_file, weights=wpth)


@assert_all
def test_with_logdir(model, config_file):
    return InferenceEngine(log_dir=ROOT, extension='.pth')


def test_with_no_input(model, config_file):
    with pytest.raises(AssertionError) as e:
        InferenceEngine()
    assert str(e.value) == 'Must provide either the log_dir or the config file'


def test_with_invalid_logdir(model, config_file):
    with pytest.raises(IndexError) as e:
        InferenceEngine(log_dir='/tmp/')
    assert str(e.value) == 'list index out of range'


def test_with_invalid_config(model, config_file):
    with pytest.raises(AssertionError) as e:
        InferenceEngine(config_file='/tmp/config.yaml')
    assert str(e.value) == 'Not Found: /tmp/config.yaml'

    with pytest.raises(AssertionError) as e:
        InferenceEngine(config_file='/tmp/')
    assert str(e.value) == 'Not Found: /tmp/'

    with pytest.raises(TypeError) as e:
        InferenceEngine(config_file=2.0)
    assert str(e.value) == 'Invalid config_file 2.0'


def test_with_input(model, config_file):
    image = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)

    ie = InferenceEngine(log_dir=ROOT, extension='.pth')
    y = ie(image)

    assert len(y.shape) == 4

    assert y.shape[0] == 1
    assert y.shape[1] == 1
    assert y.shape[2] == 224
    assert y.shape[3] == 224
