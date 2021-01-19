"""
Base Trainer for sklearn and pytorch learning tasks
"""
import abc
import pickle
import time
import typing
import numpy as np 
import tqdm
import tqdm.notebook

from .abc import AbstractTrainer, AbstractDataset, AbstractSchema
