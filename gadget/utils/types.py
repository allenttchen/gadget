#!/usr/bin/env python3

"""
Utilities for sanitary type conversion
"""
import datetime
import functools
import numpy as np 

SUPPORTED_YAML_TYPES = (
	type(None), 
	str,
	bytes,
	bool,
	int, 
	float,
	complex,
	tuple,
	list,
	set,
	dict,
	datetime.datetime
)


def sanitize_numpy_types(value):
	"""Convert numpy types to built-in python types

	Args:
		value (*): if value is a numpy type, it will be converted to its associated python type

	Returns: 
		either passed-in value or value converted to python type
	"""
	if isinstance(value, np.generic):
		return value.tolist()
	else:
		return value


def normalize_tensor_to_numpy(tensor):
	"""Normalize a tensor/vector type to numpy

		Args:
			tensor (torch.Tensor, numpy.ndarray): the tensor to be normalized to numpy
			
		Returns:
			numpy.ndarray: the input `tensor` converted to a numpy type

		Note:
			The primary reason for this function is to make functions that operate on numpy inputs appplicable to pytorch inputs as well
	"""
	try:
		import torch
		if isinstance(tensor, torch.Tensor):
			tensor = tensor.cpu().detach().numpy()
	except ModuleNotFoundError:
		pass

	return tensor


def normalize_inputs_to_numpy(func):
	"""A decorator for normalizing inputs of func to inputs of numpy types

	Example:
		.. code-block:: python

			from gadget.utils.types import normalize_inputs_to_numpy

			@normalize_inputs_to_numpy
			def sum(x, axis):
				# this example would break if the inputs were pytorch since it uses `dim` instead of `axis`.
				return x.sum(axis=axis)
	"""
	@functools.wraps(func)
	def wrapper_func(*args, **kwargs):
		args = tuple(map(normalize_tensor_to_numpy, args))
		kwargs = {key: normalize_tensor_to_numpy(value) for key, value in kwargs.items()}
		return func(*args, **kwargs)
	return wrapper_func




