"""Temporal attention with framewise queries and four-frame K/V pooling."""
import torch
import torch.nn.functional as F
from torch import nn

from pymss_core.modules.bs_roformer.transformer import apply_rotary_emb_fast, cached_rotary_cos_sin


class PooledKVAttention(nn.Module):
    heads = 2
    head_dim = 64
    stride = 4

    def __init__(self, dim, rotary, dropout):
        super().__init__()
        width = self.heads * self.head_dim
        self.query = nn.Linear(dim, width)
        self.key_value = nn.Linear(dim, 2 * width)
        self.gates = nn.Linear(dim, self.heads)
        self.output = nn.Linear(width, dim)
        self.rotary = rotary
        self.dropout = dropout

    def forward(self, x):
        batch, frames, bands, dim = x.shape
        x = x.transpose(1, 2).reshape(batch * bands, frames, dim)
        stride = self.stride
        pooled = F.avg_pool1d(x.transpose(1, 2), stride, stride, ceil_mode=True).transpose(1, 2)
        query = self.query(x).unflatten(-1, (self.heads, self.head_dim))
        key, value = self.key_value(pooled).unflatten(-1, (2, self.heads, self.head_dim)).unbind(-3)

        cos, sin = cached_rotary_cos_sin(self.rotary, frames, query.device, query.dtype)
        starts = torch.arange(0, frames, stride, device=x.device)
        centers = starts + ((frames - starts).clamp(max=stride) - 1) // 2
        query = apply_rotary_emb_fast(cos, sin, query).transpose(1, 2)
        key = apply_rotary_emb_fast(cos.index_select(1, centers), sin.index_select(1, centers), key).transpose(1, 2)
        value = value.transpose(1, 2).contiguous()
        out = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2) * self.gates(x).sigmoid().unsqueeze(-1)
        return self.output(out.flatten(-2)).reshape(batch, bands, frames, dim).transpose(1, 2).contiguous()
