import numpy as np
import numpy.fft as fft


from data.transform import mean_filter
from env import frame_size


def filtered_cepstrum(log_amp, kernel_size=20):
    low_pass_filtered = mean_filter(log_amp, kernel_size)
    hi_pass_filtered = log_amp - low_pass_filtered
    return np.abs(fft.rfft(hi_pass_filtered, axis=0))


def get_inverse_scale_dft_matrix(n):
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


def get_custom_scale_dft_matrix(full_freq, custom_freq):
    n_in, n_out = full_freq.size, custom_freq.size
    freq_per_bin = full_freq[-1] / full_freq.size
    period = custom_freq / freq_per_bin
    omega_n = np.e ** (-1j * 2 * np.pi / n_in)

    def get_vector(term):
        c = n_in / term
        return omega_n ** (np.arange(n_in) * c)

    base = np.zeros((n_out, n_in), dtype=np.complex128)
    for i in range(n_out):
        base[i, :] = get_vector(period[i])
    return base


default_inverse_scale_dft_matrix = get_inverse_scale_dft_matrix(frame_size)


def inverse_scale_cepstrum(log_amp, inv_scale_dft_mat=None):
    """
    In a cepstrum each entry corresponds to "freq" in frequency domain
    By sampling in the DTFT in a inverse manner, in the mod-cepstrum each
    entry corresponds to "period" in freq domain (to extract formant)
    :param log_amp: log amplitude
    :param inv_scale_dft_mat: inverse scale dft matrix
    :return:
    """
    if inv_scale_dft_mat is not None:
        return np.abs(np.dot(inv_scale_dft_mat, log_amp))
    n = log_amp.shape[0]
    base = get_inverse_scale_dft_matrix(n)
    return np.abs(np.dot(base, log_amp))


def mod_cepstrum(log_amp, custom_scale_dft_mat=None, filter_size=10):
    """
    High-pass filter the log_spectrum before performing
    inverse-scale cepstral analysis
    :param log_amp: log amplitude
    :param custom_scale_dft_mat: inverse scale dft matrix
    :param filter_size:
    :return:
    """
    if filter_size is None:
        return inverse_scale_cepstrum(
            log_amp, custom_scale_dft_mat).astype(log_amp.dtype)
    low_pass_filtered = mean_filter(log_amp, filter_size)
    hi_pass_filtered = log_amp - low_pass_filtered
    return inverse_scale_cepstrum(
        hi_pass_filtered, custom_scale_dft_mat).astype(log_amp.dtype)
