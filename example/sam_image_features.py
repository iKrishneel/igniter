#!/usr/bin/env python

from typing import List, Dict, Any, Iterator, Tuple
import os.path as osp
import numpy as np
import torch
from torchvision.datasets import CocoDetection as _Dataset

import hydra
from omegaconf import DictConfig

from igniter.builder import trainer
from igniter.registry import model_registry, dataset_registry, io_registry, proc_registry
from igniter.io import S3IO

from segment_anything import sam_model_registry, SamPredictor as _SamPredictor
from segment_anything.modeling import Sam


@model_registry('sam')
class SamPredictor(_SamPredictor):
    def __init__(self, *args, **kwargs):
        super(SamPredictor, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def forward(self, batched_input: List[Dict[str, Any]]) -> torch.Tensor:
        input_images = torch.stack([self.model.preprocess(x['image']) for x in batched_input], dim=0)
        return self.model.image_encoder(input_images)

    @classmethod
    def build(cls, cfg) -> Sam:
        name = cfg.build.model
        checkpoint = cfg.models[name].weights
        assert osp.isfile(checkpoint), f'Weight file not found {checkpoint}!'
        model_type = cfg.models[name].name
        sam = sam_model_registry[model_type](checkpoint=checkpoint)

        return cls(**{'sam_model': sam.to(cfg.device)})

    def children(self) -> Iterator['Module']:
        for name, module in self.model.named_children():
            yield module

    def modules(self) -> Iterator['Module']:
        for _, module in self.model.named_modules():
            yield module

    def buffers(self, *args, **kwargs):
        return self.model.buffers(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def named_modules(self, *args, **kwargs):
        return self.model.named_modules(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        return self.model.named_buffers(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)


@dataset_registry('coco')
class Dataset(_Dataset):
    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        if self.transforms is not None:
            image = self.transforms(image)

        image = torch.from_numpy(np.asarray(image).transpose((2, 0, 1)))
        return {'image': image, 'id': id}  # , 'original_size': image.shape[1:]}


@proc_registry('sam_image_feature_saver')
def sam_forward(engine, batch):
    cfg = engine._cfg
    inputs = [{'image': data['image'].to(cfg.device)} for data in batch]
    try:
        features = engine._model.module.forward(inputs)
    except AttributeError:
        features = engine._model.forward(inputs)

    for feature, data in zip(features, batch):
        id = data['id']
        fname = f'{str(int(id)).zfill(12)}'
        engine._io_ops(feature, fname)


@hydra.main(version_base=None, config_path='./configs', config_name='sam_image_features')
def main(cfg: DictConfig):
    trainer(cfg)


if __name__ == '__main__':
    main()
