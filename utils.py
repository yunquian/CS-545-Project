import numpy as np

import torch

from data.transform import db_to_power
from env import non_silent_cutoff_db


def non_silent_frames(amp, cutoff_db=non_silent_cutoff_db):
    """
    :param amp: amplitude as spectrogram
    :param cutoff_db:
    :return: (n_samples,) numpy boolean array
    """
    frame_energy = np.average(amp ** 2, axis=0)
    cutoff = np.max(frame_energy) / db_to_power(cutoff_db)
    return frame_energy > cutoff


def get_model_param_count(m: torch.nn.Module, only_trainable=True):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`

    Copied from: https://stackoverflow.com/a/62764464
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)
