"""
Implementing
"""

import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy


def basic_train():
    pass


def meta_inner_train():
    pass


def meta_reptile_train(model, n_iter_outer, n_iter_inner, outer_step_size):
    # Reptile training loop
    for iteration in range(n_iter_outer):
        weights_before = deepcopy(model.state_dict())
        # Generate task
        f = gen_task()
        y_all = f(x_all)
        # Do SGD on this task
        inds = rng.permutation(len(x_all))
        for _ in range(n_iter_inner):
            for start in range(0, len(x_all), ntrain):
                mbinds = inds[start:start + ntrain]
                train_on_batch(x_all[mbinds], y_all[mbinds])
        # Interpolate between current weights and trained weights from this task
        # I.e. (weights_before - weights_after) is the meta-gradient
        weights_after = model.state_dict()
        outerstepsize = outer_step_size * (
                1 - iteration / n_iter_outer)  # linear schedule
        model.load_state_dict(
            {name: weights_before[name] + (weights_after[name] -
                                           weights_before[name]) * outerstepsize
             for name in weights_before})
