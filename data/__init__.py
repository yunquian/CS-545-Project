import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

from data.transform import mfcc_from_amp
from env import (
    sr, n_fft, n_hop,
    n_mfcc_align, n_mfcc_model,
    non_silent_cutoff_db)
from utils import non_silent_frames


def read_audio(filename):
    """Returns: 1d numpy array"""
    fs, audio = wavfile.read(filename)
    assert fs == sr and len(audio.shape) == 1
    avg_power = np.average(audio ** 2)
    if avg_power > 1e-100:
        audio = audio / np.sqrt(avg_power)
    return audio.astype(np.float64)


class AudioData:
    def __init__(self, filename):
        audio = read_audio(filename)
        _, _, zxx = stft(audio, sr, nperseg=n_fft, noverlap=n_fft-n_hop)
        self.amp = np.abs(zxx)
        self.align_features = mfcc_from_amp(self.amp, sr, n_mfcc_align)
        self.mfcc_model = mfcc_from_amp(self.amp, sr, n_mfcc_model)
        # self.frame_energy = np.average(self.amp ** 2, axis=0)
        self.frame_energy = np.ones(self.amp.shape[1])
        # self.selected_frames = np.ones(self.amp.shape[1], dtype=np.bool)
        self.selected_frames = non_silent_frames(self.amp, non_silent_cutoff_db)

    def normalized_amp(self):
        return np.where(self.frame_energy == 0, 0,
                        self.amp / np.sqrt(self.frame_energy))
