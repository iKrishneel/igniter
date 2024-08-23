#!/usr/bin/env python

from igniter.registry import event_registry, io_registry


@event_registry('default_logging_handler')
def logging_hanlder(engine, type: str = 'tqdm', **kwargs):
    assert type in io_registry, f'{type} not found in io_registry'
    attach = kwargs.pop('attach', {})
    log_handler = io_registry[type](**kwargs)
    assert log_handler is not None
    log_handler.attach(engine, **attach)


@event_registry('default_checkpoint_handler')
def checkpoint_handler(
    engine, root: str, prefix: str = 'model_', extension: str = 'pt', save: str = 'model'
) -> None:
    writer_name = 's3_writer' if 's3://' in root else 'file_writer'
    writer = io_registry[writer_name](root=root, extension=extension)
    assert callable(writer)

    extension = extension.replace('.', '')
    filename = f'{prefix}{str(engine.state.epoch).zfill(7)}.{extension}'
    writer(engine.get_state_dict(save_options=save), filename)
