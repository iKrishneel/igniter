#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from igniter import initiate
from igniter.registry import model_registry, dataset_registry, proc_registry


@model_registry('mnist')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    @classmethod
    def build(cls, cfg):
        return cls()

    def forward(self, x, targets=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        if self.training:
            return self.losses(output, targets)
        return output

    def losses(self, x, targets):
        return {'loss': F.nll_loss(x, targets)}


@dataset_registry('mnist')
def mnist_dataset(**kwargs):
    from torchvision.datasets import MNIST

    transform = kwargs.pop('transforms')
    return MNIST(transform=transform, **kwargs)


@proc_registry('mnist_collate_fn')
def collate_fn(data):
    images, targets = [], []
    for d in data:
        images.append(d[0])
        targets.append(d[1])

    images = torch.stack(images)
    targets = torch.Tensor(targets).long()
    return images, targets


initiate('./configs/mnist.yaml')
