from random import Random
from typing import List

import numpy as np

from data import AudioData
from data.align import dtw_align
from data.metadata import Metadata
from data.transform import log_stft, db_to_amp
from data.transform.cepstrum import (
    mod_cepstrum,
    default_inverse_scale_dft_matrix)


def to_gen_model_input(amp):
    """Converts data to generative model's input"""
    return log_stft(amp) / 60


def to_gen_model_output(amp):
    """
    Converts data (amplitude of target frames) to generative model's output
    """
    return log_stft(amp) / 60


def from_gen_model_output(model_out):
    """Converts generative model's output to amplitude"""
    return db_to_amp(model_out * 60)


def audio_dat_to_log_amp_mfcc_mod_ceps(audio_data: AudioData, indices):
    if indices is None:
        indices = np.arange(audio_data.amp.shape[1])
    log_amp = log_stft(audio_data.amp)
    mfcc = audio_data.mfcc
    mod_ceps = mod_cepstrum(log_amp, default_inverse_scale_dft_matrix)
    return np.vstack([
        log_amp[:, indices] / 60,
        mfcc[1:, indices],
        mod_ceps[:mod_ceps.shape[0] // 2, indices] / 500
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
            dat1.mfcc, dat2.mfcc, (dat1.selected_frames, dat2.selected_frames))
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

    def get(self):
        alignment = dtw_align(
            self.source.mfcc, self.target.mfcc,
            (self.source.selected_frames, self.target.selected_frames))
        s_indices, t_indices = zip(*alignment)
        return (to_model_input_transform(self.source, s_indices),
                to_model_output_transform(self.target.amp[:, t_indices]))


class InputData:
    def __init__(self, filename):
        self.dat = AudioData(filename)

    def get(self):
        return to_model_input_transform(self.dat.amp, None)
