import numpy as np


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
    :param spec: amplitude
    :param mod_dft_mat:
    :return:
    """
    if mod_dft_mat is not None:
        return np.dot(mod_dft_mat, spec)
    n = spec.shape[0]
    base = get_mod_dft_matrix(n)
    return np.abs(np.dot(base, spec))