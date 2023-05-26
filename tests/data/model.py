#!/usr/bin/env python

import torch
import torch.nn as nn

from igniter.registry import model_registry


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
