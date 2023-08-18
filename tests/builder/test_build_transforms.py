#!/usr/bin/env python

import os.path as osp

import pytest
import torchvision.transforms as TF
from omegaconf import OmegaConf

from igniter.builder import build_transforms


@pytest.fixture(scope='session')
def cfg():
    config_path = osp.join(osp.dirname(osp.abspath(__file__)), '../data/configs/transforms.yaml')
    assert osp.isfile(config_path)
    return OmegaConf.load(config_path)


def test_build_transforms_only_torchvision(cfg):
    transforms = build_transforms(cfg, 'train')

    assert isinstance(transforms, TF.Compose)
    assert len(transforms.transforms) == 3

    target_transforms = [TF.ToTensor, TF.Normalize, TF.Resize]
    for transform1, transform2 in zip(transforms.transforms, target_transforms):
        assert isinstance(transform1, transform2)


def test_build_transforms_only_registry(cfg):
    transforms = build_transforms(cfg, 'val')
    assert isinstance(transforms, TF.Compose)
    assert len(transforms.transforms) == 2

    target_transforms = ['ResizeLongestSide', 'PadToSize']
    for transform in transforms.transforms:
        assert transform.__class__.__name__ in target_transforms


def test_build_transforms_both(cfg):
    transforms = build_transforms(cfg, 'test')
    assert isinstance(transforms, TF.Compose)
    assert len(transforms.transforms) == 5


def test_build_transforms_all(cfg):
    transform_dict = build_transforms(cfg)
    assert isinstance(transform_dict, dict)

    for key in ['train', 'val', 'test']:
        assert key in transform_dict
        assert isinstance(transform_dict[key], TF.Compose)


def test_build_transforms_wrong_mode(cfg):
    transform_dict = build_transforms(cfg, None)
    assert isinstance(transform_dict, dict)
    # assert len(transform_dict) == 0
