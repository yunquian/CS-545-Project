"""
Feature transform
"""
import librosa
import numpy as np


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
        np.clip(amp, max_amp / db_to_amp(dynamic_range), np.inf))
    return log_amp


def mfcc_from_amp(amp, sr, n_mfcc):
    power = np.abs(amp) ** 2
    mel_spec = librosa.feature.melspectrogram(S=power, sr=sr)
    return librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)


def to_gen_model_input(amp):
    """Converts data to generative model's input"""
    return log_stft(amp)


def to_gen_model_output(amp):
    """
    Converts data (amplitude of target frames) to generative model's output
    """
    return log_stft(amp)


def from_gen_model_output(model_out):
    """Converts generative model's output to amplitude"""
    return db_to_amp(model_out)
