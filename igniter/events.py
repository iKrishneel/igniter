#!/usr/bin/env python

from typing import Any, Dict, List
from urllib.parse import urlparse

from igniter.logger import logger
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
    state_dict = engine.get_state_dict(keys=keys)
    filename = f'{prefix}{str(engine.state.epoch).zfill(7)}.{extension}'
    has_written = checkpoint_handler_utils(state_dict, root, filename, prefix, extension, writer_name)

    if not has_written:
        io_registry['checkpoint'](engine=engine, root='./igniter_logs/', unique_dir=True)


def checkpoint_handler_utils(
    state_dict: Dict[str, Any],
    root: str,
    filename: str,
    prefix: str = 'model_',
    extension: str = 'pt',
    writer_name: str = None,
) -> bool:
    parsed = urlparse(root)
    if parsed.scheme == 's3' and writer_name is None:
        writer_name = 's3_writer'
        args = dict(bucket_name=parsed.netloc, root=parsed.path.lstrip('/'))

    if writer_name is None or writer_name == 'file_writer':
        writer_name = 'file_writer'
        args = dict(root=root, extension=extension)

    writer = io_registry[writer_name](**args)
    assert callable(writer)

    try:
        writer(state_dict, filename)
    except Exception as e:
        logger.warn(f'Failed to write state due to {e}\nBackup on the local disk')
        return False
    return True
