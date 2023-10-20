#!/usr/bin/env python

import os.path as osp
import time

from pycocotools.coco import COCO as _COCO

from ..io.s3_client import S3Client

__all__ = ['COCO']


class COCO(_COCO):
    def __init__(self, s3_client: S3Client, annotation_filename: str) -> None:
        if osp.isfile(annotation_filename):
            super(COCO, self).__init__(annotation_filename)
        else:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = s3_client(annotation_filename)
            assert type(dataset) is dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()
