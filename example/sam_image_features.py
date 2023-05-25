#!/usr/bin/env python

from typing import List, Dict, Any, Iterator, Tuple
import os.path as osp
import numpy as np
import torch
from torchvision.datasets import CocoDetection as _Dataset

from igniter.registry import model_registry, dataset_registry, io_registry, func_registry
from igniter import initiate
from igniter.io import S3IO

from segment_anything import sam_model_registry, SamPredictor as _SamPredictor
from segment_anything.modeling import Sam


@model_registry('sam')
class SamPredictor(_SamPredictor):
    def __init__(self, name: str, checkpoint: str):
        assert osp.isfile(checkpoint), f'Weight file not found {checkpoint}!'
        sam = sam_model_registry[name](checkpoint=checkpoint)
        super(SamPredictor, self).__init__(sam_model=sam)

    @torch.no_grad()
    def forward(self, batched_input: List[Dict[str, Any]]) -> torch.Tensor:
        return self.set_images([x['image'] for x in batched_input])

    @torch.no_grad()
    def set_images(self, images: List[np.ndarray], image_format: str = 'RGB') -> None:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            images = [image[..., ::-1] for image in images]

        input_images = [torch.as_tensor(self.transform.apply_image(image).transpose(2, 0, 1)) for image in images]
        input_images = [self.model.preprocess(image) for image in input_images]
        input_images_torch = torch.stack(input_images).to(self.device)
        return self.model.image_encoder(input_images_torch)

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

        image = np.asarray(image)
        return {'image': image, 'id': id}  # , 'original_size': image.shape[1:]}


@func_registry('sam_image_feature_saver')
def sam_forward(engine, batch):
    cfg = engine._cfg
    inputs = [{'image': data['image']} for data in batch]
    try:
        features = engine._model.module.forward(inputs)
    except AttributeError:
        features = engine._model.forward(inputs)

    for feature, data in zip(features, batch):
        id = data['id']
        fname = f'{str(int(id)).zfill(12)}'
        print(fname)
        engine.s3_writer(feature, fname)


initiate('./configs/sam_image_features.yaml')
