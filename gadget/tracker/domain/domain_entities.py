from argparse import Namespace
import os
from gadget.utils.dataclass_utils import validate_dataclass, field


_persistance_field_kwargs = dict(
	init=False,
	hash=False, 
	default=None, 
	repr=False,
	serializes=False
)


