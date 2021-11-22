class Metadata:
    def __init__(self, n_speakers, n_audios, mapping):
        self.n_speakers = n_speakers
        self.n_audios = n_audios
        self.mapping = mapping

    def get(self, speaker_idx, audio_idx):
        return self.mapping(speaker_idx, audio_idx)
