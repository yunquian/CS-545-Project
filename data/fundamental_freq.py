"""
Estimate fundamental frequency

"""
import numpy as np

from data.transform import mean_filter
import data.transform.cepstrum as cepstrum
import env
from env import freq

_CUTOFF_FREQ = 2000  # Do not consider freq > cutoff while cepstral analysis
_RESOLUTION = 0.3  # Hz

_N_FREQ_USED = np.sum(freq < _CUTOFF_FREQ)


def _calc_shared_constants():
    # custom sampled cepstrum
    freq_lo = env.fundamental_freq_min - 10
    freq_hi = env.fundamental_freq_max + 10
    trimmed_freq = np.linspace(
        freq_lo, freq_hi, num=int((freq_hi - freq_lo) / _RESOLUTION))
    mod_dft_mtx = cepstrum.get_custom_scale_dft_matrix(
        freq[:_N_FREQ_USED], trimmed_freq)
    return trimmed_freq, mod_dft_mtx


_TRIMMED_FREQ, _MOD_DFT_MTX = _calc_shared_constants()


class FundamentalFreqEstimator:
    def __init__(self, log_amp):
        # estimate freq
        fundamental, fundamental_energy, prob = self._extract_info(log_amp)
        self.est_prob_min = np.min(prob)
        prob -= self.est_prob_min
        self.est_prob_norm = np.sum(prob)
        prob = prob / self.est_prob_norm
        mean_freq = np.sum(fundamental*prob)
        self.average_f0 = mean_freq
        self.max_f0 = fundamental[np.argmax(prob)]
        self.best_f0 = (self.average_f0
                        if self.average_f0 + self.max_f0 > 450
                        else self.max_f0)  # hardcoded based on observation
        self.re_ets_bin_min = self._get_closest_index(
            _TRIMMED_FREQ, self.best_f0 / 1.5)
        self.re_ets_bin_max = self._get_closest_index(
            _TRIMMED_FREQ, self.best_f0 * 1.5)

    @staticmethod
    def _get_closest_index(arr, val):
        return np.argmin(np.abs(arr - val))

    def _extract_info(self, log_amp, re_estimiate=False):
        mod_ceps = cepstrum.mod_cepstrum(
            log_amp[:_N_FREQ_USED],
            custom_scale_dft_mat=_MOD_DFT_MTX, filter_size=None)
        if re_estimiate:
            fundamental_bin = self.re_ets_bin_min + np.argmax(
                mod_ceps[self.re_ets_bin_min:self.re_ets_bin_max], axis=0)
        else:
            fundamental_bin = np.argmax(mod_ceps, axis=0)
        fundamental = _TRIMMED_FREQ[fundamental_bin]
        fundamental_energy = mean_filter(mod_ceps, 3)
        prob = (np.max(fundamental_energy, axis=0)**2 /
                np.mean(fundamental_energy**2, axis=0))
        return fundamental, fundamental_energy, prob

    def estimate(self, log_amp):
        fundamental, fundamental_energy, prob = self._extract_info(
            log_amp, True)
        prob = (prob - self.est_prob_min)
        return fundamental, prob

    def get_speaker_spec(self):
        return np.array([self.average_f0,
                         self.best_f0, self.est_prob_norm])

    @staticmethod
    def transform(log_amp, source, target):
        src_fundamental, src_prob = source.estimate(log_amp)
        fundamental = (src_fundamental * target.best_f0
                       / source.best_f0)
        prob = src_prob + source.est_prob_min - target.est_prob_min
        return fundamental, prob
