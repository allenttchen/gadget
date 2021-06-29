from dataclasses import (
    Field, 
    fields, 
    asdict, 
    field as field_, 
    is_dataclass, 
    _MISSING_TYPE, 
    _is_dataclass_instance, 
)
import types
import typing
import functools


def field(
    *args, 
    options: typing.Union[set, list, tuple] = None, 
    serializes: bool = True, 
    required: bool = False, 
    help: str = "", 
    **kwargs, 
) -> Field:
    
    metadata = dict(serializes=serializes, required=required, help=help)
    if options is not None:
        metadata["options"] = options
    if required:
        kwargs["default"] = None
    return field_(*args, metadata=metadata, **kwargs)
