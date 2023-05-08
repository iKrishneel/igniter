#!/usr/bin/env pythono

from typing import List, Dict, Any

import torch.nn as nn
from torch.utils.data import DataLoader

import importlib
import os

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, BasicTimeProfiler
from ignite.contrib.handlers import ProgressBar
import ignite.distributed as idist

from igniter.registry import model_registry, dataset_registry, io_registry, proc_registry
from igniter.utils import is_distributed
from igniter.logger import logger


def build_transforms(cfg) -> Dict[str, Any]:
    transforms = {}
    for key in cfg.transforms:
        module = importlib.import_module(cfg.transforms[key].engine)
        augs = cfg.transforms[key].augs
        if augs is None:
            transforms[key] = None
            continue
        transforms[key] = module.Compose([getattr(module, cls)(**(augs[cls]) if augs[cls] else {}) for cls in augs])
    return transforms


def collate_fn(data):
    return data


def build_train_dataloader(cfg) -> Dict[str, DataLoader]:
    name = cfg.build.dataset
    cls = dataset_registry[name]
    attrs = cfg.datasets
    transforms = build_transforms(cfg)
    dataloaders = {}
    for key in attrs[name]:
        if key not in attrs[name]:
            continue
        try:
            logger.info(f'Building {key} dataloader')
            dataset = cls(**{**dict(attrs[name][key]), 'transforms': transforms.get(key, None)})
            dataloaders[key] = DataLoader(dataset, collate_fn=collate_fn, **dict(cfg.datasets.dataloader))
        except TypeError:
            dataloaders[key] = None
    return dataloaders


def build_model(cfg) -> nn.Module:
    logger.info(f'Building network model {cfg.build.model}')
    cls_or_func = model_registry[cfg.build.model]
    try:
        return cls_or_func.build(cfg)
    except AttributeError:
        return cls_or_func(cfg)


def build_optim(cfg, model):
    module = importlib.import_module(cfg.solvers.engine)
    name = cfg.build.solver
    return getattr(module, name)(model.parameters(), **cfg.solvers[name])


def add_profiler(engine, cfg):
    profiler = BasicTimeProfiler()
    profiler.attach(engine)

    @engine.on(Events.ITERATION_COMPLETED(every=cfg.snapshot))
    def log_intermediate_results():
        profiler.print_results(profiler.get_results())

    return profiler


def build_io(cfg):
    engine = cfg.io.engine
    cls = io_registry[engine]
    if cls is None:
        cls = importlib.import_module(engine)
    try:
        return cls.build(cfg)
    except AttributeError:
        return None


def build_func(cfg):
    func = proc_registry[cfg.build.func]
    if func is None:
        logger.info('Using default training function')
        func = proc_registry['default']
    assert func, f'Training forward function is not defined'
    return func


class TrainerEngine(Engine):
    def __init__(
        self, cfg, process_func, model, optimizer, dataloaders: Dict[str, DataLoader], io_ops=None, **kwargs
    ) -> None:
        super(TrainerEngine, self).__init__(process_func, **kwargs)

        # TODO: Move this to each builder function
        if is_distributed(cfg):
            model = idist.auto_model(model)
            optimizer = idist.auto_optim(optimizer)
            for key in dataloaders:
                if dataloaders[key] is None:
                    continue
                dataloaders[key] = idist.auto_dataloader(
                    dataloaders[key].dataset_registry, collate_fn=collate_fn, **dict(cfg.datasets.dataloader)
                )

        self._cfg = cfg
        self._model = model
        self._optimizer = optimizer
        self._train_dl, self._val = dataloaders['train'], dataloaders['val']
        self._io_ops = io_ops

        self.checkpoint()
        self.add_progress_bar()

    @classmethod
    def build(cls, cfg) -> 'TrainerEngine':
        os.makedirs(cfg.workdir, exist_ok=True)
        model = build_model(cfg)
        optimizer = build_optim(cfg, model)
        dls = build_train_dataloader(cfg)
        io_ops = build_io(cfg)
        update_model = build_func(cfg)
        return cls(cfg, update_model, model, optimizer, dataloaders=dls, io_ops=io_ops)

    def __call__(self):
        self.run(self._train_dl, self._cfg.epochs, epoch_length=len(self._train_dl))

    def checkpoint(self):
        if self._cfg.snapshot == 0:
            return
        self.add_event_handler(
            Events.ITERATION_COMPLETED(every=self._cfg.snapshot) | Events.EPOCH_COMPLETED,
            Checkpoint({'model': self._model, 'optimizer': self._optimizer}, self._cfg.workdir, n_saved=2),
        )

    def add_progress_bar(self, output_transform=lambda x: {"loss": x}):
        ProgressBar(persist=True).attach(self, output_transform=output_transform)


def build_trainer(cfg):
    return TrainerEngine.build(cfg)


def _trainer(rank, cfg):
    trainer = build_trainer(cfg)
    trainer()


def trainer(cfg):
    if is_distributed(cfg):
        init_args = dict(cfg.distributed[cfg.distributed.type])
        with idist.Parallel(backend=cfg.distributed.backend, **init_args) as parallel:
            parallel.run(_trainer, cfg)
    else:
        _trainer(None, cfg)
