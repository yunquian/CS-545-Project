import torch.nn as nn
from torch.nn import MSELoss
import torchaudio

from env import sr, frame_size
from data.transform import db_to_amp


class MelMSE(nn.Module):
    def __init__(self, n_mel=128, is_input_log_amp=True):
        super(MelMSE, self).__init__()
        self.mel = torchaudio.transforms.MelScale(
            sample_rate=sr, n_stft=frame_size, n_mels=n_mel)
        self.mse = MSELoss()
        self.is_input_log_amp = is_input_log_amp

    def forward(self, model_output, target):
        if self.is_input_log_amp:
            model_output = db_to_amp(model_output)
            target = db_to_amp(target)
        return self.mse(self.mel(model_output), self.mel(target))
