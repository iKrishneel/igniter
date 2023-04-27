#!/usr/bin/env pythono

from typing import List, Dict, Any, Tuple, Iterator
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from torchvision.datasets import CocoDetection as _Dataset

import importlib
import os
import os.path as osp

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, BasicTimeProfiler
from ignite.contrib.handlers import ProgressBar
import ignite.distributed as idist

from segment_anything import sam_model_registry, SamPredictor as _SamPredictor

from igniter.registry import model_registry, dataset_registry
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


class TrainerEngine(Engine):
    def __init__(self, cfg, process_func, model, optimizer, dataloaders: Dict[str, DataLoader], **kwargs):
        super(TrainerEngine, self).__init__(process_func, **kwargs)

        self._cfg = cfg
        self._model = model
        self._optimizer = optimizer
        self._train_dl, self._val = dataloaders['train'], dataloaders['val']

        self.checkpoint()
        self.add_progress_bar()

    @classmethod
    def build(cls, cfg) -> 'TrainerEngine':
        os.makedirs(cfg.workdir, exist_ok=True)

        model = build_model(cfg)
        optimizer = build_optim(cfg, model)

        dls = build_train_dataloader(cfg)

        if is_distributed(cfg):
            model = idist.auto_model(model)
            optimizer = idist.auto_optim(optimizer)
            for key in dls:
                if dls[key] is None:
                    continue
                dls[key] = idist.auto_dataloader(
                    dls[key].dataset_registry, collate_fn=collate_fn, **dict(cfg.datasets.dataloader)
                )

        def update_model(engine, batch):
            inputs = [{'image': data['image'].to(cfg.device)} for data in batch]

            try:
                features = model.module.forward(inputs)
            except AttributeError:
                features = model.forward(inputs)

            for feature, data in zip(features, batch):
                id = data['id']
                torch.save(feature, osp.join(cfg.workdir, f"{str(int(id)).zfill(12)}.pt"))
            """
            self._model.train()
            inputs, targets = batch
            self._optimizer.zero_grad()
            loss = model(inputs)
            # loss = criterion(logits, outputs)
            loss.backward()
            self._optimizer.step()
            return loss.item()
            """

        return cls(cfg, update_model, model, optimizer, dataloaders=dls)

    def __call__(self):
        super().run(self._train_dl, self._cfg.epochs, epoch_length=len(self._train_dl))

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


def trainer(cfg):
    def _trainer(rank, cfg):
        trainer = build_trainer(cfg)
        trainer()

    if is_distributed(cfg):
        init_args = dict(cfg.distributed[cfg.distributed.type])
        with idist.Parallel(backend=cfg.distributed.backend, **init_args) as parallel:
            parallel.run(_trainer, cfg)
    else:
        _trainer(None, cfg)
