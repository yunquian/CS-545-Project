"""
This file includes global parameters
"""

# input param
sr = 16000  # all audios has sr = 16000

# fft
n_fft = 2048
n_hop = 512
frame_size = n_fft // 2 + 1

# mfcc
n_mfcc_align = 20  # used for alignment
n_mfcc_model = 128  # used for generative model

# amplitude cutoff
non_silent_cutoff_db = 30

# fundamental freq
# https://en.wikipedia.org/wiki/Voice_frequency
fundamental_freq_min = 85
fundamental_freq_max = 255

freq = None
