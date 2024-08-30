#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

from tabulate import tabulate

from igniter.logger import logger


@dataclass
class Registry(object):
    name: str = 'REGISTRY'
    overwrite: bool = False
    strict: bool = True
    __REGISTRY: Dict[str, object] = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.overwrite and self.strict:
            logger.warning('Both stict and overwrite flags are set. Not strict will take precedence!')

    def __call__(self, name_or_cls: Union[str, object] = None, prefix: Optional[str] = None):
        def _wrapper(cls):
            assert callable(cls)
            name = cls.__name__ if name_or_cls is None or not isinstance(name_or_cls, str) else name_or_cls
            name = prefix + name if prefix else name
            if name in self.__REGISTRY:
                if self.strict:
                    raise ValueError(f'{cls} is already registered {self.__REGISTRY}')

                if not self.overwrite:
                    return
                else:
                    logger.warning(f'⚠️  Overwriting {name} ...')
            self.__REGISTRY[name] = cls
            return cls

        if callable(name_or_cls):
            name = str(name_or_cls.__name__)  # NOQA: F841
            return _wrapper(name_or_cls)
        return _wrapper

    def __getitem__(self, name: str):
        if name not in self.__REGISTRY:
            print(self)
            raise KeyError(f'Key {name} not found in the {self.name}')
        return self.__REGISTRY.get(name)

    def __contains__(self, key: str):
        return key in self.__REGISTRY.keys()

    def register(self, name_or_cls: Union[str, object] = None):
        return self(name_or_cls=name_or_cls)

    def get(self, name: str) -> Callable:
        return self[name] if name in self else None

    def remove(self, name: str) -> Any:
        logger.warning(f'⚠️  Removing {name}')        
        return self.__REGISTRY.pop(name, None)

    def deregister(self, name: str) -> Any:
        return self.remove(name)

    def clear(self) -> None:
        if self.strict:
            logger.warning('⚠️  Cannot clear registry with strict flag set')
            return

        logger.warning(f'⚠️  Clearning {self.name}')
        self.__REGISTRY.clear()

    def __repr__(self):
        title = f'Registry for {self.name}\n'
        return title + tabulate(
            [[k, v] for k, v in self.__REGISTRY.items()], headers=['Name', 'Objects'], tablefmt="fancy_grid"
        )


engine_registry = Registry(name='Engine Registry')
model_registry = Registry(name='Model Registry')
dataset_registry = Registry(name='Dataset Registry')
solver_registry = Registry(name='Solver Registry')
io_registry = Registry(name='IO Registry')
func_registry = Registry(name='Function Registry')
transform_registry = Registry(name='Transform Registry')
event_registry = Registry(name="Event Handlers Registry")

# for backward compatibility
proc_registry = func_registry
