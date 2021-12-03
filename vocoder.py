"""
Vocoder: converts spectrogram to audio

Available vocoders are Griffin-Lim and WaveNet(not implemented)
"""
import librosa

from env import n_fft, n_hop


def griffin_lim(amp):
    return librosa.feature.inverse.griffinlim(
        amp, hop_length=n_hop, win_length=n_fft,
        n_iter=500)
