import torch
import torch.nn as nn
from torch.nn import Linear


class ToyVAE(nn.Module):
    def __init__(self, frame_size=1024):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
