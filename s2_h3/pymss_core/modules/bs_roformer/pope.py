# Adapted from https://github.com/lucidrains/PoPE-pytorch (f306e51).
# MIT License
# Copyright (c) 2026 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from math import pi
import torch
import torch.nn.functional as F
from torch import nn


class PoPE(nn.Module):
    """Polar positional encoding with the BS PolarFormer checkpoint layout."""
    def __init__(self, dim, *, heads, theta=10000):
        super().__init__()
        self.register_buffer("inv_freqs", theta ** -(torch.arange(dim).float() / dim))
        self.bias = nn.Parameter(torch.zeros(heads, dim))

    def forward(self, q, k):
        # Keep positions precise even when assign=True loads FP16 checkpoint buffers.
        dtype = torch.float64 if self.inv_freqs.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=q.device.type, enabled=False):
            positions = torch.arange(k.shape[-2], device=q.device, dtype=dtype)
            freqs = positions[:, None] * self.inv_freqs.to(dtype)[None, :]
            key_freqs = freqs + self.bias.to(dtype).clamp(-2 * pi, 0)[:, None, :]
            query_freqs = freqs[-q.shape[-2]:]
            q_magnitude, k_magnitude = F.softplus(q), F.softplus(k)
            q_polar = torch.stack((q_magnitude * query_freqs.cos(), q_magnitude * query_freqs.sin()), dim=-1).flatten(-2)
            k_polar = torch.stack((k_magnitude * key_freqs.cos(), k_magnitude * key_freqs.sin()), dim=-1).flatten(-2)
        return q_polar.to(q.dtype), k_polar.to(k.dtype)
