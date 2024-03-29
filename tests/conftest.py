#!/usr/bin/env python

import os
import os.path as osp
import shutil

import pytest
import torch
from data.datasets import *  # NOQA
from data.model import ExampleModel
from data.transforms import *  # NOQA
from omegaconf import OmegaConf

ROOT: str = '/tmp/igniter/tests'
os.makedirs(ROOT, exist_ok=True)


def clean_up():
    shutil.rmtree(ROOT)


@pytest.fixture(scope='session', autouse=True)
def config_file():
    config_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data/configs/config.yaml')
    return config_path


@pytest.fixture(scope='session', autouse=True)
def model(config_file, request):
    request.addfinalizer(clean_up)
    config = OmegaConf.load(config_file)

    wpth = config.build[config.build.model].weights
    os.makedirs(osp.dirname(wpth), exist_ok=True)

    model = ExampleModel()
    torch.save({'model': model.state_dict()}, wpth)
    shutil.copy(config_file, osp.join(ROOT, 'config.yaml'))
    return model
