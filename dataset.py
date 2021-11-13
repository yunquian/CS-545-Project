import numpy as np

from scipy.signal import stft
from scipy.io import wavfile

from align import dtw_align
from env import sr, n_fft
from utils import db_to_amp


def read_audio(filename):
    fs, audio = wavfile.read(filename)
    assert fs == sr and len(audio.shape) == 1
    return audio


def raw_dataset():
    pass


class MetaDataset:
    """

    """

    def __init__(self):
        pass

    def create_samples(self, i, j):
        pass
