"""
This file includes global parameters
"""
import numpy as np

# input param

sr = 16000  # all audios has sr = 16000

# fft
n_fft = 1024
# n_fft = 4096
n_hop = 128
frame_size = n_fft // 2 + 1

# mfcc
n_mfcc_align = 64  # used for alignment
n_mfcc_model = 128  # used for generative model

# amplitude cutoff
non_silent_cutoff_db = 20

# fundamental freq
# https://en.wikipedia.org/wiki/Voice_frequency
fundamental_freq_min = 85
fundamental_freq_max = 255

freq = np.linspace(0, sr/2, num=frame_size, endpoint=True)

freq_per_bin = sr / 2 / (frame_size - 1)
