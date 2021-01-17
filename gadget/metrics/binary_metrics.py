#!/usr/bin/env python3

import warnings
import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support
from gadget.utils.types import normalize_inputs_to_numpy


@normalize_inputs_to_numpy
def binary_classification_metrics_at_threshold(y_true, y_prob, threshold=0.5, pos_label=1):
	"""Compute sklearn binary classification metrics with the ability to set a pre-defined threshold
		Args:
			y_true (numpy.array int): vector of true class labels represented by ints 0 and 1
			y_prob (numpy.array float): vector of probability predictions [0, 1] values
			threshold (float): probability threshold, above which a prediction will be considered class label pos_label
			pos_label (int): 0 or 1; Integer label of the "positive" class, of which probability predictions in y_prob are associated

		Returns:
			dict: mapping between metric keys and their values
	"""

	# Turn probability labels into integer labels, with probs > threshold resulting in class label = pos_label
	y_label = (y_prob > threshold) * 1
	# Flip integer label if positive label is 0
	if pos_label == 0:
		y_pred = 1 - y_pred

	# Calculate metrics and confusion matrix with sklearn
	precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=pos_label, average='binary')

	# Confusion Matrix
	if pos_label == 0:
		y_true = 1 - y_true
		y_pred = 1 - y_pred
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

	# Support: count of pos_label occurences in y_true
	support = tp + fn

	# Accuracy: Percent of correct prediction out of all data
	if (tn + fp + fn + tp > 0):
		accuracy = (tp + tn) / ((tn + fp + fn + tp) * 1.0)
	else:
		accuracy = np.nan

	# False Positive Rate
	if (fp + fn > 0):
		false_positive_rate = fp / (fp + tn)
	else:
		false_positive_rate = np.nan

	return {
		"precision": precision, 
		"recall": recall, 
		"f1": f1, 
		"false_positive_rate": false_positive_rate, 
		"accuracy": accuracy, 
		"support": support, 
		"TN": tn, 
		"FP": fp, 
		"FN": fn, 
		"TP": tp,
		"threshold": threshold
	}


@normalize_inputs_to_numpy
def find_threshold_at_recall_target(y_true, y_prob, recall_target=0.90, pos_label=1):
	"""Uses Sklearn function precision_recall_curve to find the threshold that gives the smallest recall, such that recall >= recall_target, for two classes. If there is no recall >= recall_target, nans is returned

		Args:
			y_true (numpy.array int): vector of true class labels represented by ints 0 and 1
			y_prob (numpy.array float): vector of probability predictions [0, 1] values
			recall_target (float): recall targeted, at which to set threshold, wrt pos_label class
			pos_label (int): 0 or 1; Integer label of the "positive" class, of which the probability predictions in y_prob are associated

		Returns:
			float: the classification threshold which obtains :code:'recall=recall_target'
	"""
	precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob, pos_label=pos_label)

	# Find index that has the smallest recall such that recall >= target_recall
	# Assumes recall is already sorted, greatest to least (default sklearn function behavior)
	idx = np.argwhere(recalls >= recall_target)
	if len(idx) > 0:
		idx = idx.max()
		threshold = thresholds[idx]
	else:
		threshold = np.nan

	return threshold


@normalize_inputs_to_numpy
def binary_classification_metrics_at_recall_target(y_true, y_prob, recall_target=0.9, pos_label=1, failure_action="error"):
	"""Compute Sklearn binary classification metrics at a given recall target
		Args:
			y_true (numpy.array int): vector of true class labels represented by ints 0 and 1
			y_prob (numpy.array float): vector of probability predictions [0, 1] values
			recall_target (float): recall targeted, at which to set threshold, wrt pos_label class
			pos_label (int): 0 or 1; Integer label of the "positive" class, of which the probability predictions in y_prob are associated
			failure_action (str): [options=("error", "warn", "ignore")] This option controls the behavior in the casses that there is no threshold which can achieve :code:`recall=recall_target`. 
				The default case is to raise an error (as per standard python policy), but this can be changed to either issue a warning or to ignore completely. 
				If :code:`failure_action=="warn"` or :code:`failure_action=="ignore"`, the values in the returned dictionary will be all np.nan.

		Returns:
			dict: mapping between metric keys and their values
	"""

	# Find the threshold that gives the smallest recall such that recall >= recall_target
	threshold = find_threshold_at_recall_target(y_true, y_prob, recall_target, pos_label)
	# subtract an epsilon to account for floating point error in comparison
	threshold = threshold - np.finfo(float).eps

	if not np.isnan(threshold):
		metrics_dict = binary_classification_metrics_at_threshold(y_true, y_prob, threshold, pos_label)
	elif failure_action == "error":
		raise RuntimeError(
			"The function 'binary_classification_metrics_at_recall_target' returned a null result, which "
			"usually indicates that the data sample is irregular (e.g. all non-target class data points) "
			"and no suitable classification threhsold was found to achieve the recall target. \n"
			"Possible remediations: \n"
			"  1. Use 'binary_classification_metrics_at_threshold' and pick a threshold. \n"
			"  2. Set the argument to this function, 'failure_action' to 'warn' or 'ignore'. \n"
		)
	else:
		if failure_action == "warn":
			warnings.warn(
				"The function 'binary_classification_metrics_at_recall_target' returned a null result, which "
				"usually indicates that the data sample is irregular (e.g. all non-target class data points) "
				"and no suitable classification threhsold was found to achieve the recall target. \n"
				"Possible remediations: \n"
				"  1. Use 'binary_classification_metrics_at_threshold' and pick a threshold. \n"
				"  2. Set the argument to this function, 'failure_action' to 'warn' or 'ignore'. \n"
			)
		metrics_dict = {
			"precision": np.nan, 
			"recall": np.nan, 
			"f1": np.nan, 
			"false_positive_rate": np.nan, 
			"accuracy": np.nan, 
			"support": np.nan, 
			"TN": np.nan, 
			"FP": np.nan, 
			"FN": np.nan, 
			"TP": np.nan, 
			"threshold": np.nan
		}

	return metrics_dict










