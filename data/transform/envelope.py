"""
The function `calc_tae` is adapted from `calc_true_envelope_spectral` in
https://github.com/SubramaniKrishna/VaPar-Synth/blob/master/extra_dependencies/func_envs.py
, whose author is Krishna Subramani
"""


import numpy as np

import env


def calc_tae(log_amp, thresh=0.1, stopping_iters=10):
	"""
	calculates True Amplitude Envelope
	:param log_amp: (n_freq, n_time) or (n_freq,) numpy array for log amplitude
	:param thresh: threshold
	:param stopping_iters: max iterations
	:return: TAE, cepstrum_envelope
	"""
	if len(log_amp.shape) == 1:
		log_amp.reshape((-1, 1))
	# cutoff
	max_fundamental_freq = env.fundamental_freq_max + 10
	bin_cutoff = int(env.sr/(2*max_fundamental_freq))
	# ceps envelope
	ceps = np.fft.rfft(log_amp, axis=0).copy()
	ceps[bin_cutoff:] = 0
	lo = np.fft.irfft(ceps, axis=0)
	ceps_env = log_amp.copy()
	s = min(ceps_env.shape[0], lo.shape[0])
	ceps_env[:s] = lo[:s]
	# start TAE
	A_ip1 = log_amp
	A_0 = A_ip1
	# Threshold array
	thresh_arr = thresh * np.ones_like(log_amp)
	cou = 0
	while True:
		ceps = np.fft.rfft(A_ip1, axis=0).copy()
		ceps[bin_cutoff:] = 0
		lo = np.fft.irfft(ceps, axis=0)
		V_i = A_ip1.copy()
		s = min(V_i.shape[0], lo.shape[0])
		V_i[:s] = lo[:s]
		A_ip1 = np.where((A_ip1 > V_i), A_ip1, V_i)
		cou = cou + 1
		# Stopping Criteria
		if np.all(((A_0 - V_i) <= thresh_arr)) or (cou >= stopping_iters):
			Vf = V_i
			break
	return Vf, ceps_env








