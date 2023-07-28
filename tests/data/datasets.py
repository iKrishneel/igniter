#!/usr/bin/env python

from torch.utils.data import Dataset

from igniter.registry import dataset_registry


@dataset_registry('test_dataset')
class TestDataset(Dataset):
    def __init__(self, root, *args, **kwargs):
        super(TestDataset, self).__init__()

    def __len__(self):
        return 5
