from typing import List
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from env import *
from data import AudioData
from data.align import dtw_align
from data.metadata import Metadata
from data.transform import dft_filter, log_stft


class FormantCoderDataset:
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

    def __getitem__(self, index):
        amp = self.all_audios[index].amp
        lo, hi = dft_filter(log_stft(amp))
        return lo


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
            dat1.mfcc_align, dat2.mfcc_align,
            (dat1.selected_frames, dat2.selected_frames))
        if k is None:
            selected_index_pairs = alignment
        else:
            selected_index_pairs = self.rng.sample(alignment, k)
        s1_indices, s2_indices = zip(*selected_index_pairs)
        return dat1, s1_indices, dat2, s2_indices


class SplitModelTaskDataset:
    def __init__(self, source_filename, target_filename):
        self.source = AudioData(source_filename)
        self.target = AudioData(target_filename)

    def get_as_aligned_amp(self):
        alignment = dtw_align(
            self.source.mfcc_align, self.target.mfcc_align,
            (self.source.selected_frames, self.target.selected_frames))
        s_indices, t_indices = zip(*alignment)
        return self.source.amp[:, s_indices], self.target.amp[:, t_indices]

    def get(self):
        alignment = dtw_align(
            self.source.mfcc_align, self.target.mfcc_align,
            (self.source.selected_frames, self.target.selected_frames))
        s_indices, t_indices = zip(*alignment)
        return self.source, s_indices, self.target, t_indices


class FormantCoder(nn.Module):
    """
    Formant coder encodes and decodes formant information
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1,
                      kernel_size=(8,), stride=(4,)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=1, out_channels=3,
                      kernel_size=(8,), stride=(4,)),
            nn.LeakyReLU(negative_slope=0.01),
        )
        #
        # self.formant_transform = nn.Sequential(
        #      nn.Linear(125*5 + self.n_mfcc, 123*5),
        # )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=3, out_channels=1,
                               output_padding=(3,),
                               kernel_size=(8,), stride=(4,)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose1d(in_channels=1, out_channels=1,
                               output_padding=(1,),
                               kernel_size=(8,), stride=(4,))
        )

    def encode(self, x):
        x = x.view(x.shape[0], 1, x.shape[1]) / 60
        return self.encoder(x)

    def decode(self, x):
        x = self.decoder(x)
        return x.view(x.shape[0], x.shape[2]) * 60

    def forward(self, x):
        return self.decode(self.encode(x))

    def test_dimension(self):
        x = torch.zeros((3, 1025))
        print('encoded shape', self.encode(x).shape)
        print('decoded shape', self.forward(x).shape)


class FormantTransformer(nn.Module):
    """
    Formant Transformer transforms the embedding
    """
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(3*62, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3*62),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        n_samples, n_channels, n = x.shape
        return self.transform(
            x.view(n_samples, n_channels*n)).view(n_samples, n_channels, n)


class FormantModel:
    pass


class SplitModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input
        self.frame_size = frame_size
        self.n_mfcc = n_mfcc_model - 1
        # self.n_mod_ceps = frame_size // 2 + 1
        self.n_mod_ceps = 85

        # model

        self.formant_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1,
                      kernel_size=(16,), stride=(2,)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=1, out_channels=1,
                      kernel_size=(8,), stride=(4,)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(125, 123),
            nn.ConvTranspose1d(in_channels=1, out_channels=1,
                               output_padding=(1,),
                               kernel_size=(8,), stride=(2,)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose1d(in_channels=1, out_channels=1,
                               output_padding=(1,),
                               kernel_size=(16,), stride=(4,))
        )

        self.fundamental_freq_model = nn.Sequential(
            nn.Linear(self.n_mod_ceps + self.n_mfcc, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, frame_size),
        )
        # self.model = nn.Sequential(
        #     nn.Linear(frame_size + n_mfcc_model - 1 + frame_size // 4, 128),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.Linear(128, 256),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.Linear(256, frame_size),
        # )

    def forward(self, x):
        # split
        log_stft = x[:, :frame_size]
        mfcc = x[:, frame_size:frame_size + self.n_mfcc]
        fundamental_freq_dat = x[:, frame_size + self.n_mfcc
                        :frame_size + self.n_mfcc + self.n_mod_ceps]
        filtered_formant = x[:, frame_size + self.n_mfcc + self.n_mod_ceps:]
        # print(self.formant_model(
        #     log_stft.view(-1, 1, frame_size)).shape)

        ceps = self.fundamental_freq_model
        return torch.cat(
            (
             self.formant_model(filtered_formant.view(
                 -1, 1, frame_size)).view(-1, frame_size),
             self.fundamental_freq_model(torch.cat((fundamental_freq_dat, mfcc), 1))
            ), 1)


class SplitLoss(nn.Module):
    pass


if __name__ == '__main__':
    FormantCoder().test_dimension()
