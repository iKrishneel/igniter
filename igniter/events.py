#!/usr/bin/env python

from typing import List
from urllib.parse import urlparse

from igniter.registry import event_registry, io_registry


@event_registry('default_checkpoint_handler')
def checkpoint_handler(
    engine,
    root: str,
    prefix: str = 'model_',
    extension: str = 'pt',
    writer_name: str = None,
    keys: List[str] = 'all',
) -> None:
    if 's3://' in root and writer_name is None:
        writer_name = 's3_writer'

        parsed = urlparse(root)
        args = dict(bucket_name=parsed.netloc, root=parsed.path.lstrip('/'))

    if writer_name is None or writer_name == 'file_writer':
        writer_name = 'file_writer'
        args = dict(root=root, extension=extension)

    writer = io_registry[writer_name](**args)
    assert callable(writer)

    extension = extension.replace('.', '')
    filename = f'{prefix}{str(engine.state.epoch).zfill(7)}.{extension}'
    writer(engine.get_state_dict(keys=keys), filename)
