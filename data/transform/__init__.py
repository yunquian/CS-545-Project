"""
Feature transform
"""
import numpy as np
from numpy.fft import rfft, irfft
from scipy import signal

import librosa

from env import *


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


def log_stft(amp, dynamic_range=120):
    """
    :param amp: amplitude
    :param dynamic_range: in db
    :return: time (1D array), frequencies (1D array), log amplitude
        log_amp is (num_freq, num_time) numpy array, in decibels

    Note on dynamic_range:
    -------
    Assuming that the weakest detectable sound is 0 db, then the loudest
    possible sound is 194 db.

    db = 20 * log_10(amp)
    """
    max_amp = np.max(amp)
    if max_amp == 0:
        return amp
    log_amp = 20 * np.log10(
        np.clip(amp, max_amp / db_to_amp(dynamic_range), np.inf))+dynamic_range
    return log_amp


def mfcc_from_amp(amp, sr, n_mfcc):
    power = np.abs(amp) ** 2
    mel_spec = librosa.feature.melspectrogram(S=power, sr=sr)
    return librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)


def mean_filter(spec, kernel_size=10, zero_pad=False):
    if len(spec.shape) == 1:
        kernel = np.ones(kernel_size) / kernel_size
    else:
        kernel = np.ones((kernel_size, 1)) / kernel_size
    ret = signal.convolve(spec, kernel, mode='same')
    if not zero_pad:
        ret[:kernel_size//2] = ret[:kernel_size//2]
        ret[-kernel_size//2:] = ret[-kernel_size//2]
    return ret


def dft_filter(log_amp):
    """
    :param log_amp:
    :return: (spectral envelope, fundamental freq & harmonics)
    """
    c_ceps = rfft(log_amp, axis=0)
    lo_ceps = c_ceps.copy()
    lo_ceps[20:] = 0  # empirical, sr/2/env.fundamental_freq_max - 11
    lo = np.zeros_like(log_amp)
    lo_recon = irfft(lo_ceps, axis=0)
    lo[:lo_recon.shape[0]] = lo_recon

    hi_ceps = c_ceps.copy()
    hi_ceps[:20] = 0
    hi_ceps[105:] = 0  # empirical, sr/2/env.fundamental_freq_min + 11
    hi = np.zeros_like(log_amp)
    hi_recon = irfft(hi_ceps, axis=0)
    hi[:hi_recon.shape[0]] = hi_recon

    return lo, hi
