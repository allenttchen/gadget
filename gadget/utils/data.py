#!/usr/bin/env python3

"""
Utilities for datasets
"""

import numpy as np
import pandas as pd 
from collections import Counter


def assign_folds(df, num_folds, label_column_name=None, seed=42):
	"""Assign fold indices to the data points in a dataframe

		Args:
			df (pandas.DataFrame): a dataframe containing the dataset
			num_folds (int): the number of folds to create in the dataset
			label_column_name (str): [default=None] the name of the column containing the supervision label.
				The dataset will be stratified by this label to keep the class distribution the same in each fold.
				If the default, None, is used, it will be assumed there is no label upon which to stratified the folds (such as in unsupervised settings)
			seed (int): [default=42] the fold assignmnet is deterministic for a given seed.

		Returns:
			pandas.DataFrame: a copy of `df` with a new column `fold_index`
	"""

	if label_column_name is not None:
		# partitioning and keep label imbalance proportions
		partitions = {label: [] for label in df[label_column_name].unique()}
		for _, row in df.iterrows():
			partitions[row[label_column_name]].append(row.to_dict())
	else:
		# using a list of row dicts to match the format above
		partitions = {"": [row.to_dict() for _, row in df.iterrows()]}

	np.random.seed(seed)

	output = []
	# Splitting within each class
	for label, data_list in partitions.items():
		np.random.shuffle(data_list)
		n_data = len(data_list)
		# Number of records per fold
		per_fold = int(n_data / num_folds)

		# Assign fold index to each record in the data_list
		for fold_index in range(num_folds):
			start_index = fold_index * per_fold
			end_index = (fold_index + 1) * per_fold

			for data_record in data_list[start_index:end_index]:
				data_record["fold_index"] = fold_index

		# assign fold index to the rest of the record sequentially
		for counter, record in enumerate(data_list[end_index:]):
			data_record["fold_index"] = counter % num_folds

		output.extend(data_list)

	return pd.DataFrame(output)


