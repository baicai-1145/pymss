"""Frequency-axis ablations from the 62-band MG experiment."""
import torch
import torch.nn.functional as F
from torch import nn


class BandMixer(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm
        self.weight = nn.Parameter(torch.randn(8, 62, 62) / 62 ** 0.5)

    def set_cuda_attention_backend(self, backend=None):
        pass

    def set_mps_attention_backend(self, backend=None, min_tokens=128):
        pass

    def forward(self, x):
        grouped = self.norm(x).unflatten(-1, (8, 24))
        return torch.einsum('gof,bfgc->bogc', self.weight, grouped).flatten(-2)


class BandFFN(nn.Module):
    def __init__(self, old):
        super().__init__()
        self.norm, self.dropout, self.out_dropout = old.net[0], old.net[3], old.net[5]
        self.up, self.down = nn.Linear(192, 1024), nn.Linear(512, 192)

    def forward(self, x):
        u, v = self.up(self.norm(x)).chunk(2, -1)
        return self.out_dropout(self.down(self.dropout(u * F.silu(v))))
