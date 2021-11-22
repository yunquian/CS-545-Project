import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

from data.transform import mfcc_from_amp
from env import sr, n_fft, n_mfcc


def read_audio(filename):
    fs, audio = wavfile.read(filename)
    assert fs == sr and len(audio.shape) == 1
    avg_power = np.average(audio ** 2)
    if avg_power > 1e-100:
        audio = audio / np.sqrt(avg_power)
    return audio.astype(np.float32)


class AudioData:
    def __init__(self, filename):
        audio = read_audio(filename)
        _, _, zxx = stft(audio, sr, nperseg=n_fft)
        self.amp = np.abs(zxx)
        self.mfcc = mfcc_from_amp(self.amp, sr, n_mfcc)
        self.selected_frames = np.ones(self.amp.shape[1], dtype=np.bool)
        # non_silent_frames(self.amp, non_silent_cutoff_db)
