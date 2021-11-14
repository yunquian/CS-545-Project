import torch.nn as nn
import torch.nn.functional as F

from env import n_fft


class ToyVAE(nn.Module):
    def __init__(self, frame_size=n_fft):
        super().__init__()
        self.fc1 = nn.Linear(frame_size, 256)
        self.fc2 = nn.Linear(256, frame_size)

    def forward(self, x):
        return self.fc2(F.leaky_relu(self.fc1(x), negative_slope=0.2))
