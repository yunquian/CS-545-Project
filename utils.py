import numpy as np
import torch


def db_to_amp(db):
    """
    Converts decibels to amplitude
    db = 20 * log10(amp / amp_ref)
    :param db: decibels
    :return: amplitude
    """
    return 10.0 ** (db / 20)


def db_to_power(db):
    """
    Converts decibels to power
    power_db = 10 * log10(amp / amp_ref)
    :param db: decibels
    :return: power
    """
    return 10.0 ** (db / 10)


def non_silent_frames(amp, cutoff_db=60):
    """
    :param amp: amplitude as spectrogram
    :param cutoff_db:
    :return: (n_samples,) numpy boolean array
    """
    frame_energy = np.average(amp ** 2, axis=0)
    cutoff = np.max(frame_energy) / db_to_power(cutoff_db)
    return frame_energy > cutoff


def save_model(path, model):
    torch.save(model.state_dict(), path)


def load_model(path, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
