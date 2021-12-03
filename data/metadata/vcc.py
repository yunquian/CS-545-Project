from data.metadata import Metadata


def train_2016(speaker=0, audio=1):
    assert 0 <= audio < 162
    speakers = ['SF1', 'SF2', 'SF3', 'SM1', 'SM2',
                'TF1', 'TF2', 'TM1', 'TM2', 'TM3']
    return ('datasets/vcc2016/vcc2016_training/' + speakers[speaker] + '/'
            + str(100001 + audio) + '.wav')


def train_2016_appended(speaker=0, audio=1):
    assert 0 <= audio < 108
    speakers = ['SF1', 'SF2', 'SF3', 'SM1', 'SM2',
                'TF1', 'TF2', 'TM1', 'TM2', 'TM3']
    return ('datasets/vcc2016/vcc2016_training_appended/' + speakers[speaker]
            + '/' + str(100001 + audio) + '.wav')


def test_2016(speaker=0, audio=1):
    raise NotImplementedError


def train_2018(speaker=0, audio=0):
    assert 0 <= audio < 81
    speakers = ['SF1', 'SF2', 'SF3', 'SF4',
                'SM1', 'SM2', 'SM3', 'SM4',
                'TF1', 'TF2', 'TM1', 'TM2']
    return ('datasets/vcc2018/vcc2018_training/VCC2' + speakers[speaker] + '/'
            + str(10001 + audio) + '.wav')


def test_2018(speaker=0, audio=1):
    raise NotImplementedError


class SanityMetadata(Metadata):
    def __init__(self):
        super(SanityMetadata, self).__init__(
            n_speakers=4, n_audios=4, mapping=train_2016)


class VCC2016TrainMetadata(Metadata):
    def __init__(self):
        super().__init__(n_speakers=10, n_audios=162, mapping=train_2016)


class VCC2016TrainAppendedMetadata(Metadata):
    def __init__(self):
        super().__init__(n_speakers=10, n_audios=108,
                         mapping=train_2016_appended)
