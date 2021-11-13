def train_2016(speaker=0, audio=1):
    assert 0 <= audio <= 161
    speakers = ['SF1', 'SF2', 'SF3', 'SM1', 'SM2',
                'TF1', 'TF2', 'TM1', 'TM2', 'TM3']
    return ('datasets/vcc2016/vcc2016_training/' + speakers[speaker] + '/'
            + str(100001 + audio) + '.wav')


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


class MetaData:
    def __init__(self, n_speakers, n_audios, mapping):
        self.n_speakers = n_speakers
        self.n_audios = n_audios
        self.n_mapping = mapping

    def