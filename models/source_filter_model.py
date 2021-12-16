from typing import List
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import *
from data import AudioData
from data.align import dtw_align
from data.metadata import Metadata
from data.transform import dft_filter, log_stft
from data.transform.envelope import calc_tae
from data.fundamental_freq import FundamentalFreqEstimator


class AutoEncoderDataset:
    def __init__(self, metadata: Metadata):
        self.all_audio_filenames = [
            metadata.get(speaker_id, audio_id)
            for speaker_id in range(metadata.n_speakers)
            for audio_id in range(metadata.n_audios)
        ]
        self.n_audios = len(self.all_audio_filenames)
        self.all_audios = [
            AudioData(filename)
            for filename in self.all_audio_filenames
        ]
        self.parsed_dataset = []
        for index in range(self.n_audios):
            selected_frames = self.all_audios[index].selected_frames
            amp = self.all_audios[index].normalized_amp()[:, selected_frames]
            tae, ceps_env = calc_tae(log_stft(amp))
            self.parsed_dataset.append((tae, ceps_env))

    def __getitem__(self, index):
        tae, ceps_env = self.parsed_dataset[index]
        return tae, ceps_env


class SplitModelMetaDataset:
    def __init__(self, metadata: Metadata, random_seed=0):
        self.metadata = metadata
        self.rng = random.Random(random_seed)
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
        return dat1, s1_indices, dat2, s2_indices


class SourceFilterTaskDataset:
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
        amp_s, amp_t = self.get_as_aligned_amp()
        log_amp_s, log_amp_t = log_stft(amp_s), log_stft(amp_t)
        tae_s, ceps_env_s = calc_tae(log_amp_s)
        tae_t, ceps_env_t = calc_tae(log_amp_t)
        return [tae_s, ceps_env_s], [tae_t, ceps_env_t]


class EnvelopeAE(nn.Module):
    """
    Formant coder encodes and decodes formant information
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3,
                      kernel_size=(16,), stride=(8,)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=3, out_channels=5,
                      kernel_size=(8,), stride=(4,)),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=5, out_channels=3,
                               output_padding=(3,),
                               kernel_size=(8,), stride=(4,)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose1d(in_channels=3, out_channels=1,
                               output_padding=(1,),
                               kernel_size=(16,), stride=(8,))
        )

    def encode(self, x):
        x = x.view(x.shape[0], 1, x.shape[1]) / 100
        return self.encoder(x)

    def decode(self, x):
        x = self.decoder(x)
        return x.view(x.shape[0], x.shape[2]) * 100

    def forward(self, x):
        return self.decode(self.encode(x))

    def test_dimension(self):
        x = torch.zeros((3, frame_size))
        print('encoded shape', self.encode(x).shape)
        print('decoded shape', self.forward(x).shape)


class EnvelopeTransformer(nn.Module):
    """
    Formant Transformer transforms the embedding
    """
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(5*14, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 5*14),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        n_samples, n_channels, n = x.shape
        return self.transform(
            x.view(n_samples, n_channels*n)).view(n_samples, n_channels, n)


class Reconstructor:
    def __init__(self):
        self.positional_coding = np.linspace(
            0, sr / 2, num=frame_size, endpoint=True).reshape(
            (-1, 1))

    def get_log_amp(self, true_amp_env, ceps_env, f0, prob=None):
        f0 = f0.reshape((1, -1))
        harmonics = np.cos(self.positional_coding * 2 * np.pi / f0)
        env_hi = np.maximum(true_amp_env, ceps_env)
        env_center = np.minimum(true_amp_env, ceps_env)
        harmonics_amp = env_hi - env_center
        log_amp = harmonics * harmonics_amp + env_center
        return log_amp


if __name__ == '__main__':
    EnvelopeAE().test_dimension()
