"""
Feature transform
"""
import numpy as np
import librosa

from utils import db_to_amp


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


def get_mod_dft_matrix(n):
    """
    Returns a (n, n) mod-dft matrix s.t. each row corresponds to period
        instead of freq
    :param n:
    :return:
    """
    omega_n = np.e ** (-1j * 2 * np.pi / n)

    def get_vector(term):
        if term == 0:
            c = 0
        else:
            c = n / term
        return omega_n ** (np.arange(n) * c)

    base = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        base[i, :] = get_vector(i)
    return base


def mod_cepstrum(spec, mod_dft_mat=None):
    """
    In a cepstrum each entry corresponds to "freq" in frequency domain
    By sampling in the DTFT in a inverse manner, in the mod-cepstrum each
    entry corresponds to "period" in freq domain (to extract formant)
    :param spec:
    :param mod_dft_mat:
    :return:
    """
    if mod_dft_mat is not None:
        return np.dot(mod_dft_mat, spec)
    n = spec.shape[0]
    base = get_mod_dft_matrix(n)
    return np.abs(np.dot(base, spec))
