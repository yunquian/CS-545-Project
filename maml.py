"""
Implementing
"""

import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy

from dataset import MetaDataset


def basic_train(model, x, y):
    pass


def reptile_train(model, meta_dataset: MetaDataset, n_shot,
                  n_iter_outer, outer_step_size, inner_train_func):
    # Reptile training loop
    for iteration in range(n_iter_outer):
        weights_before = deepcopy(model.state_dict())
        # Generate task
        x, y = meta_dataset.sample(n_shot)
        # Do SGD on this task
        inner_train_func(model, x, y)
        # Interpolate between current weights and trained weights from this task
        # I.e. (weights_before - weights_after) is the meta-gradient
        weights_after = model.state_dict()
        outerstepsize = outer_step_size * (
                1 - iteration / n_iter_outer)  # linear schedule
        model.load_state_dict(
            {name: weights_before[name] + (weights_after[name] -
                                           weights_before[name]) * outerstepsize
             for name in weights_before})
