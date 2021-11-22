"""
Implementing MAML
"""

import torch
from copy import deepcopy

from data.dataset import MetaDataset


def _inner_train_template(model, x, y, n_iter=1000,
                          log_period=None):
    """
    :param model: model to be trained
    :param x: (n_feature, n) tensor as model input
    :param y: (n_feature, n) tensor as target output
    :param n_iter: number of iteration
    :param log_period: period in iterations of logging, None for don't log
    """
    pass


def reptile_train(model, meta_dataset: MetaDataset, n_shot,
                  n_iter_meta, meta_step_size,
                  inner_train_func, n_iter_inner=1000,
                  log_period_meta=10, log_period_inner=250):
    # Reptile training loop
    for iteration in range(n_iter_meta):
        should_log = (log_period_meta is not None
                      and iteration % log_period_meta == 0)
        inner_log_period = log_period_inner if should_log else None
        weights_before = deepcopy(model.state_dict())
        # Generate task
        x, y = meta_dataset.sample(n_shot)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # Do optimization on this task
        if should_log:
            print('Meta iter', iteration, ': ')
        inner_train_func(model, x, y, n_iter=n_iter_inner,
                         log_period=inner_log_period)
        # Interpolate between current weights and trained weights from this task
        # I.e. (weights_before - weights_after) is the meta-gradient
        weights_after = model.state_dict()
        step_size = meta_step_size * (
                1 - iteration / n_iter_meta)  # linear schedule
        model.load_state_dict(
            {name: weights_before[name] + (weights_after[name] -
                                           weights_before[name]) * step_size
             for name in weights_before})
