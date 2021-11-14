from random import Random
from typing import List

from scipy.signal import stft
from scipy.io import wavfile

from align import dtw_align
from env import sr, n_fft, n_mfcc, non_silent_cutoff_db
from metadata import Metadata
from transform import mfcc_from_amp
from utils import db_to_amp, non_silent_frames


def read_audio(filename):
    fs, audio = wavfile.read(filename)
    assert fs == sr and len(audio.shape) == 1
    return audio


class AlignedAudioData:
    def __init__(self, filename):
        audio = read_audio(filename)
        self.amp = stft(audio, sr, nperseg=n_fft)
        self.mfcc = mfcc_from_amp(self.amp, sr, n_mfcc)
        self.selected_frames = non_silent_frames(self.amp, non_silent_cutoff_db)


class MetaDataset:
    def __init__(self, metadata: Metadata, random_seed=0):
        self.metadata = metadata
        self.rng = Random(random_seed)
        self.dat: List[List[AlignedAudioData]] = []

    def read_and_preprocess(self):
        for i in range(self.metadata.n_speakers):
            audio_data = []
            for j in range(self.metadata.n_audios):
                audio_data.append(AlignedAudioData(self.metadata.get(i, j)))
            self.dat.append(audio_data)

    def sample(self, k, is_speaker_different=True):
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
        selected_index_pairs = self.rng.sample(alignment, k)
        s1_indices, s2_indices = zip(*selected_index_pairs)
        return dat1.amp[:, s1_indices], dat2.amp[:, s2_indices]
