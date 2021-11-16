import torch.nn as nn
import torch.nn.functional as F

from env import n_fft


class ToyVAE(nn.Module):
    def __init__(self, frame_size=n_fft // 2 + 1):
        super().__init__()
        self.fc1 = nn.Linear(frame_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, frame_size)

    def forward(self, x):
        return self.fc3(
            F.leaky_relu(self.fc2(
                F.leaky_relu(self.fc1(x), negative_slope=0.2))))
