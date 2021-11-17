"""
This file includes global parameters
"""

sr = 16000  # all audios has sr = 16000

n_fft = 2048
n_mfcc = 20  # used for alignment
non_silent_cutoff_db = 40

frame_size = n_fft // 2 + 1

freq = None
