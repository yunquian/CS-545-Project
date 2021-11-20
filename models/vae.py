import torch.nn as nn
import torch.nn.functional as F
import torch

from env import n_fft


class ToyVAE(nn.Module):
    def __init__(self, frame_size=n_fft // 2 + 1):
        super().__init__()
        self.fc1 = nn.Linear(frame_size, 512)
        self.mu = nn.Linear(512, 512)
        self.log_var = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, frame_size)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.mu(h1), self.log_var(h1)

    def decode(self, z):
        h2 = F.relu(self.fc2(z))
        return self.sigmoid(self.fc3(h2))

    def calculate(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, n_fft // 2 + 1))
        z = self.calculate(mu, log_var)
        return self.decode(z), mu, log_var