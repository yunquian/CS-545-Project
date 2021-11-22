from random import Random
from typing import List

from data import AudioData
from data.align import dtw_align
from data.metadata import Metadata
from data.transform import to_gen_model_input, to_gen_model_output

to_model_input_transform = to_gen_model_input
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
        :param k:
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
        return (to_model_input_transform(dat1.amp[:, s1_indices]),
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
        return (to_model_input_transform(self.source.amp[:, s_indices]),
                to_model_output_transform(self.target.amp[:, t_indices]))


class InputData:
    def __init__(self, filename):
        self.dat = AudioData(filename)

    def get(self):
        return to_model_input_transform(self.dat.amp)
