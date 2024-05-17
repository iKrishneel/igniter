#!/usr/bin/env python

import functools
from typing import Any, Callable, Dict

from torch.utils.data import Dataset as _Dataset


def apply_transforms(func: Callable):
    @functools.wraps(func)
    def wrapper(self, index: int):
        output = func(self, index)
        output = self._transforms(output)

        import IPython; IPython.embed()
        
        return output
    return wrapper


class Dataset(_Dataset):
    def __init__(self, transforms, **kwargs: Dict[str, Any]):
        super(Dataset, self).__init__()
        self._transforms = transforms

    @apply_transforms
    def __getitem__(self, index: int):
        # raise NotImplementedError('__getitem__() not yet implemented')
        print("Ok")

    def __len__(self) -> int:
        raise NotImplementedError('Not yet implemented')


@transform_registry
class MyData(Dataset):
    def __init__(self, x  = 1):
        super().__init__(x)

    def __getitem__(self, index):
        print(f"Got index: {index}")
