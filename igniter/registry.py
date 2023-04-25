#!/usr/bin/env python

#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Dict, Union

from tabulate import tabulate


@dataclass
class Registry(object):
    name: str = 'REGISTRY'
    __REGISTRY: Dict[str, object] = field(default_factory=lambda: {})

    def __call__(self, name_or_cls: Union[str, object] = None):
        def _wrapper(cls):
            assert callable(cls)
            name = cls.__name__ if name_or_cls is None or not isinstance(name_or_cls, str) else name_or_cls
            if name in self.__REGISTRY:
                raise ValueError(f'{cls} is already registered {self.__REGISTRY}')
            self.__REGISTRY[name] = cls
            return cls

        if callable(name_or_cls):
            name = str(name_or_cls.__name__)  # NOQA: F841
            return _wrapper(name_or_cls)
        return _wrapper

    def __getitem__(self, name: str):
        return self.__REGISTRY[name]

    def register(self, name_or_cls: Union[str, object] = None):
        return self(name_or_cls=name_or_cls)

    def get(self, name: str):
        return self[name]

    def __repr__(self):
        title = f'Registry for {self.name}\n'
        return title + tabulate(
            [[k, v] for k, v in self.__REGISTRY.items()], headers=['Name', 'Objects'], tablefmt="fancy_grid"
        )


model_registry = Registry(name='Model Registry')
dataset_registry = Registry(name='Dataset Registry')
solver_registry = Registry(name='Solver Registry')
