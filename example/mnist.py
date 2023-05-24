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
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, targets=None):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)

        if targets is not None:
            losses = self.losses(x, targets)
            if self.training:
                return losses
            return x, losses

        return x

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


@proc_registry('accuracy')
def metric(engine, name):
    from ignite.metrics import Accuracy

    def _output_transform(data):
        data['y'] = data.pop('y_true')
        return data

    Accuracy(output_transform=_output_transform).attach(engine, name)


initiate('./configs/mnist.yaml')
