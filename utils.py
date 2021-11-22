import numpy as np

from data.transform import db_to_power


def non_silent_frames(amp, cutoff_db=60):
    """
    :param amp: amplitude as spectrogram
    :param cutoff_db:
    :return: (n_samples,) numpy boolean array
    """
    frame_energy = np.average(amp ** 2, axis=0)
    cutoff = np.max(frame_energy) / db_to_power(cutoff_db)
    return frame_energy > cutoff
