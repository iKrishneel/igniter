#!/usr/bin/env python

import time
from typing import Dict, Any
from pycocotools.coco import COCO as _COCO

from ..io.s3_client import S3Client

__all__ = ['COCO']


class COCO(_COCO):
    def __init__(self, s3_client, annotation_filename: str):
        print('loading annotations into memory...')
        tic = time.time()
        dataset = s3_client(annotation_filename)
        assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))
        self.dataset = dataset
        self.createIndex()
