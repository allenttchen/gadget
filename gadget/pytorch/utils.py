#!/usr/bin/python3

import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader


def make_sort_trim_collate_func(sequence_length_key, 
								sort_batch=False,
								trim_batch=False,
								trim_batch_targets=None):
	""" This makes a collate function that optionally sorts or trims the batch by the provided key
	Args:
		sequence_length_key (str): the key in the batch dictionary which corresponds to the lengths of sequences that will be used to sort or trim the batch
		sort_batch (bool): [default=False] if True, the batch will be sorted by `sequence_length_key`, requiring `sequence_length_key` be provided
		trim_batch (bool): [default=False] if True, the batch will be trimmed to the longest length in the `sequence_length_key` vector
		trim_batch_targets (None, tuple): The set of tensors to trim along the sequence dimension; 
			It is assumed that `dim=1` for these targets is the sequence dimension.
	Returns:
		function: the collate function which uses sort_key to sort the batch
	"""
	trim_batch_targets = trim_batch_targets or tuple()

	def collate_func(batch):
		batch = default_collate(batch)
		trim_index = batch[sequence_length_key].max()
		keys = batch[sequence_length_key].detach().numpy()
		sorted_keys = keys.argsort()[::-1].tolist()
		batch_out = {}
		for name, tensor in batch.items():
			if sort_batch:
				tensor = tensor[sorted_keys]
			if trim_batch and name in trim_batch_targets:
				tensor = tensor[:, :trim_index]
			batch_out[name] = tensor
		return batch_out
	return collate_func


def generate_batches(dataset, 
					 batch_size, 
					 shuffle=True,
					 drop_last=True,
					 device="cpu", 
					 sequence_length_key=None, 
					 sort_batch=False,
					 trim_batch=False,
					 trim_batch_targets=None,
					 dataloader_kwargs=None):
	""" Generates batches from a dataset in the form of an iterator

	Args: 
		dataset (torch.utils.data.Dataset): the instantiated dataset
		batch_size (int): the size of the batches
		shuffle (bool): [default=True] batches are formed from shuffled indices
		drop_last (bool): [default=True] do not return the final batch if it's smaller than the specified batch size
		device (str): [default="cpu"] the device to move the tensors to
		sequence_length_key (None or str): [default=None] The key should point to a vector of scalars that represent the length of each item in the batch
		sort_batch (bool): [default=False] if True, the batch will be sorted by `sequence_length_key`; Requires `sequence_length_key` be provided
		trim_batch (bool): [default=False] if True, the batch will be trimmed to the longest length in the `sequence_length_key` vector
		trim_batch_targets (None, tuple): The set of tensors to trim along the sequence dimension; 
			It is assumed that `dim=1` for these targets is the sequence dimension.
		dataloader_kwargs (dict or None): [default=None] Any additional arguments to the DataLoader can be specified

	Returns:
		dict: a dictionary mapping from tensor name to tensor object where the first dimension of tensor object is the batch dimension

	Note:
		This function is mostly an iterator for the DataLoader, but has the added features that 
			(1) it moves the tensors to a target device,
			(2) can sort the batch by sequence length, and
			(3) can trim the batch by the longest sequence in the batch.
	"""
	dataloader_kwargs = dataloader_kwargs or {}
	
	if sequence_length_key is None:
		collate_fn = default_collate
	else:
		collate_fn = make_sort_trim_collate_func(sequence_length_key, 
												 sort_batch=sort_batch,
												 trim_batch=trim_batch,
												 trim_batch_targets=trim_batch_targets)
	dataloader = DataLoader(dataset=dataset,
							batch_size=batch_size, 
							shuffle=shuffle, 
							drop_last=drop_last, 
							collate_fn=collate_fn, 
							**dataloader_kwargs)

	for data_dict in dataloader:
		out_data_dict = {}
		for name, tensor in data_dict.items():
			out_data_dict[name] = data_dict[name].to(device)
		yield out_data_dict

























