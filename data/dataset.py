"""
This file is part of the discarded attempts
"""

from random import Random
from typing import List

import numpy as np
import torch
from scipy import signal

from data import AudioData
from data.align import dtw_align
from data.metadata import Metadata
from data.transform import log_stft, db_to_amp, mean_filter, dft_filter
from data.transform.cepstrum import (
    mod_cepstrum, filtered_cepstrum,
    default_inverse_scale_dft_matrix)
from env import *


def to_gen_model_input(amp):
    """Converts data to generative model's input"""
    return log_stft(amp) / 60


def to_gen_model_output(amp):
    """
    Converts data (amplitude of target frames) to generative model's output
    """
    log_amp = log_stft(amp)
    # filter_size = 61
    # lo = mean_filter(log_amp, kernel_size=filter_size)
    # hi = log_amp - lo
    # # filter hi
    # mask = 1 / (1 + np.e ** (np.linspace(-6, 6, num=frame_size)))
    # mask = np.ones(frame_size)
    # hi = hi * mask.reshape((-1, 1))
    lo, hi = dft_filter(log_amp)
    return np.vstack([
        lo, hi]).astype(np.float32) / 60


def to_gen_model_output_old(amp):
    return log_stft(amp).astype(np.float32) / 60


def from_gen_model_output(model_out):
    """Converts generative model's output to amplitude"""
    print(model_out.shape)
    log_amp = torch.sum(model_out.reshape(-1, 2, frame_size), dim=1) * 60
    return db_to_amp(log_amp)


def audio_dat_to_log_amp_mfcc_mod_ceps(audio_data: AudioData, indices):
    if indices is None:
        indices = np.arange(audio_data.amp.shape[1])
    log_amp = log_stft(audio_data.amp[:, indices])
    mfcc = audio_data.mfcc_model[1:, indices]
    # filter_size = 61
    # mod_ceps = np.sqrt(mod_cepstrum(
    #     log_amp, default_inverse_scale_dft_matrix, filter_size=filter_size))
    # ceps = filtered_cepstrum(log_amp, kernel_size=filter_size)
    # envelope = mean_filter(log_amp, kernel_size=filter_size) / 60

    c_ceps = np.fft.rfft(log_amp, axis=0)
    lo_ceps = c_ceps.copy()
    lo_ceps[20:] = 0  # empirical, sr/2/env.fundamental_freq_max - 11
    envelope = np.zeros_like(log_amp)
    envelope[:frame_size-1] = np.fft.irfft(lo_ceps, axis=0)
    # envelope = np.fft.irfft(lo_ceps, axis=0)

    hi_ceps = c_ceps.copy()
    hi_ceps[:20] = 0
    hi_ceps[105:] = 0  # empirical, sr/2/env.fundamental_freq_min + 11
    hi_recon = np.fft.irfft(hi_ceps, axis=0)
    ceps_fundamental_freq = np.abs(c_ceps[20:105])

    # envelope, fundamental_freq = dft_filter(log_amp)
    # ceps = np.fft.rfft(fundamental_freq)

    # scaling
    return np.vstack([
        log_amp / 60,
        mfcc / 300,
        ceps_fundamental_freq / 4000,
        envelope / 60
    ]).astype(np.float32)


to_model_input_transform = audio_dat_to_log_amp_mfcc_mod_ceps
to_model_output_transform = to_gen_model_output


class MetaDataset:
    def __init__(self, metadata: Metadata, random_seed=0):
        self.metadata = metadata
        self.rng = Random(random_seed)
        self.dat: List[List[AudioData]] = []

    def read_and_preprocess(self):
        for i in range(self.metadata.n_speakers):
            audio_data = []
            for j in range(self.metadata.n_audios):
                audio_data.append(AudioData(self.metadata.get(i, j)))
            self.dat.append(audio_data)

    def sample(self, k=None, is_speaker_different=True):
        """
        Selects a random task and then draws k samples from the task
        :param k: k shot, None for all samples
        :param is_speaker_different:
        :return:
        """
        s1 = self.rng.randrange(self.metadata.n_speakers)
        if is_speaker_different:
            s2 = self.rng.randrange(self.metadata.n_speakers)
        else:
            s2 = self.rng.randrange(self.metadata.n_speakers - 1)
            if s2 >= s1:
                s2 += 1
        audio_index = self.rng.randrange(self.metadata.n_audios)
        dat1, dat2 = self.dat[s1][audio_index], self.dat[s2][audio_index]
        alignment = dtw_align(
            dat1.align_features, dat2.align_features,
            (dat1.selected_frames, dat2.selected_frames))
        if k is None:
            selected_index_pairs = alignment
        else:
            selected_index_pairs = self.rng.sample(alignment, k)
        s1_indices, s2_indices = zip(*selected_index_pairs)
        return (to_model_input_transform(dat1, s1_indices),
                to_model_output_transform(dat2.amp[:, s2_indices]))


class TaskDataset:
    def __init__(self, source_filename, target_filename):
        self.source = AudioData(source_filename)
        self.target = AudioData(target_filename)

    def get_as_aligned_amp(self):
        alignment = dtw_align(
            self.source.align_features, self.target.align_features,
            (self.source.selected_frames, self.target.selected_frames))
        s_indices, t_indices = zip(*alignment)
        return self.source.amp[:, s_indices], self.target.amp[:, t_indices]

    def get(self):
        alignment = dtw_align(
            self.source.align_features, self.target.align_features,
            (self.source.selected_frames, self.target.selected_frames))
        s_indices, t_indices = zip(*alignment)
        return (to_model_input_transform(self.source, s_indices),
                to_model_output_transform(self.target.amp[:, t_indices]))


class InputData:
    def __init__(self, filename):
        self.dat = AudioData(filename)

    def get(self):
        return to_model_input_transform(self.dat, None)
