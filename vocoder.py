"""
Vocoder: converts spectrogram to audio

Available vocoders are Griffin-Lim and WaveNet(not implemented)
"""
import librosa

from env import n_fft


def griffin_lim(amp):
    return librosa.feature.inverse.griffinlim(
        amp, hop_length=n_fft // 2, win_length=n_fft,
        n_iter=500)
