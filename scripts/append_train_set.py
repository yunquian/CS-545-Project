"""
This file appends raw dataset together so that each audio is > ? seconds

P.S. when running this script, make sure the current working dir is root
"""
import os

import numpy as np
from scipy.io.wavfile import write
from data import read_audio
from data.metadata import Metadata
from env import sr


def _read_all(metadata: Metadata):
    """
    :param metadata:
    :return:
    """
    ret = []
    for speaker_id in range(metadata.n_speakers):
        audios = []
        for audio_id in range(metadata.n_audios):
            audios.append(read_audio(metadata.get(speaker_id, audio_id)))
        ret.append(audios)
    return ret


class _TargetMetadata:
    def save_audio(self, speaker_id, audio_id, audio):
        pass


def _append_data(metadata: Metadata, target_metadata: _TargetMetadata,
                min_time=3.0):
    dat = _read_all(metadata)
    # calc mean time
    mean_time = []
    for audio_id in range(metadata.n_audios):
        total_time = 0
        for speaker_id in range(metadata.n_speakers):
            total_time += dat[speaker_id][audio_id].size / sr
        mean_time.append(total_time / metadata.n_speakers)
    # def metadata
    start = 0
    current_write_id = 0
    current_sum = 0
    for audio_id in range(metadata.n_audios):
        current_sum += mean_time[audio_id]
        if current_sum >= min_time:
            # save
            for speaker_id in range(metadata.n_speakers):
                target_metadata.save_audio(
                    speaker_id, current_write_id,
                    np.concatenate(dat[speaker_id][start: audio_id + 1]))
            # reset
            start = audio_id + 1
            current_sum = 0
            current_write_id += 1


class VCC2016TrainTargetMetadata(_TargetMetadata):
    def __init__(self):
        self.speakers = [
            'SF1', 'SF2', 'SF3', 'SM1', 'SM2',
            'TF1', 'TF2', 'TM1', 'TM2', 'TM3']

    def save_audio(self, speaker_id, audio_id, audio):
        path = ('datasets/vcc2016/vcc2016_training_appended/' +
                self.speakers[speaker_id])
        os.makedirs(path, exist_ok=True)
        write(path + '/' + str(100001 + audio_id) + '.wav', sr, audio)


if __name__ == '__main__':
    from data.metadata.vcc import VCC2016TrainMetadata
    _append_data(VCC2016TrainMetadata(), VCC2016TrainTargetMetadata(), 3)
