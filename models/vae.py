import torch.nn as nn
import torch.nn.functional as F

from env import n_fft


class ToyVAE(nn.Module):
    def __init__(self, frame_size=n_fft // 2 + 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(frame_size, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, frame_size),
        )

    def forward(self, x):
        return self.model(x)


class MultiFeaturePerceptron(nn.Module):
    def __init__(self, frame_size=n_fft // 2 + 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(frame_size+19+frame_size//2, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, frame_size),
        )

    def forward(self, x):
        return self.model(x)
