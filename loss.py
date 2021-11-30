import math

import numpy as np
import scipy.ndimage

import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
import torchaudio

from env import sr, frame_size
from data.transform import db_to_amp


class MSE(nn.Module):
    def __init__(self, is_input_log_amp=True):
        super().__init__()
        self.is_input_log_amp = is_input_log_amp

    def forward(self, model_output, target):
        if self.is_input_log_amp:
            model_output = db_to_amp(model_output) ** 2
            target = db_to_amp(target) ** 2
        diff = model_output - target
        return (diff**2).mean()


class MelMSE(nn.Module):
    def __init__(self, n_mel=128, is_input_log_amp=True):
        super().__init__()
        self.mel = torchaudio.transforms.MelScale(
            sample_rate=sr, n_stft=frame_size, n_mels=n_mel)
        self.mse = MSELoss()
        self.is_input_log_amp = is_input_log_amp

    def forward(self, model_output, target):
        if self.is_input_log_amp:
            model_output = db_to_amp(model_output)
            target = db_to_amp(target)
        return self.mse(self.mel(model_output), self.mel(target))


class MelFilteredMSE(nn.Module):
    """
    Before performing the MSE, the data is transformed with a varying size
    filter, the size of filter varies in mel scale
    """
    def __init__(self):
        super().__init__()
        self.transform_matrix = None
        self.is_input_log_amp = True

    def transform(self, spec):
        return torch.matmul(self.transform_matrix, spec)

    def forward(self, model_output, target):
        if self.is_input_log_amp:
            model_output = db_to_amp(model_output) ** 2
            target = db_to_amp(target) ** 2
        diff = model_output - target
        return (self.transform(diff)**2).mean()

    @staticmethod
    def to_mel(freq):
        return 2595 * np.log10(1 + freq / 700)

    @staticmethod
    def to_freq(mel):
        return 700 * (10 ** (mel / 2595) - 1)


class MelMeanFilteredMSE(MelFilteredMSE):
    def __init__(self, max_filter_size=20):
        super().__init__()
        max_freq = sr / 2
        freq = np.linspace(0, max_freq, num=frame_size, endpoint=True)
        mel = self.to_mel(freq)
        min_mel, max_mel = mel[0], mel[-1]
        raw_filter_size = ((mel - min_mel) / (max_mel - min_mel)
                           * (max_filter_size - 1) + 1)
        mtx = np.zeros((frame_size, frame_size), dtype=np.float32)
        for i in range(frame_size):
            filter_size = math.floor(raw_filter_size[i]) // 2 * 2 + 1
            start, end = i - filter_size // 2, i + filter_size // 2 + 1
            start, end = max(0, start), min(end, frame_size)
            mtx[i, start:end] = 1 / (end - start)
        self.transform_matrix = nn.Parameter(torch.from_numpy(mtx))


class MelGaussianFilteredMSE(MelFilteredMSE):
    def __init__(self, max_filter_size=20):
        super().__init__()
        max_freq = sr / 2
        freq = np.linspace(0, max_freq, num=frame_size, endpoint=True)
        mel = self.to_mel(freq)
        min_mel, max_mel = mel[0], mel[-1]
        raw_filter_size = ((mel - min_mel) / (max_mel - min_mel)
                           * (max_filter_size - 1) + 1)
        mtx = np.zeros((frame_size, frame_size))
        for i in range(frame_size):
            filter_size = math.floor(raw_filter_size[i]) // 2 * 2 + 1
            if filter_size == 1:
                mtx[i, i] = 1
                continue
            unit_impulse = np.zeros(filter_size)
            unit_impulse[filter_size//2] = 1
            sigma = (filter_size - 1) / 6
            f = scipy.ndimage.gaussian_filter1d(unit_impulse, sigma)
            # range check
            start, end = i - filter_size // 2, i + filter_size // 2 + 1
            start, end = max(0, start), min(end, frame_size)
            f_start = start - i + filter_size // 2
            f_end = end - i + filter_size // 2
            f = f[f_start:f_end]
            mtx[i, start:end] = f / np.sum(f)
        self.transform_matrix = nn.Parameter(
            torch.from_numpy(mtx.astype(np.float32)))
