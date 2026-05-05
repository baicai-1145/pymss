import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange

from .attend import Attend


def rotate_half(x):
    out = torch.empty_like(x)
    out[..., ::2] = -x[..., 1::2]
    out[..., 1::2] = x[..., ::2]
    return out


def apply_rotary_emb_fast(cos, sin, t):
    return (t * cos) + (rotate_half(t) * sin)


def cached_rotary_cos_sin(rotary_embed, seq_len, device, dtype):
    cache = getattr(rotary_embed, '_pymss_cos_sin_cache', None)
    if cache is None:
        cache = {}
        rotary_embed._pymss_cos_sin_cache = cache

    key = (seq_len, device.type, device.index, dtype)
    cached = cache.get(key)
    if cached is not None:
        return cached

    freqs = rotary_embed.forward(
        lambda: rotary_embed.get_seq_pos(seq_len, device=device, dtype=dtype, offset=0),
        cache_key=f'freqs:{seq_len}|offset:0'
    )
    freqs = rearrange(freqs, 'n d -> 1 n 1 d').to(device=device, dtype=dtype)
    cached = (freqs.cos(), freqs.sin())
    cache[key] = cached
    return cached


def rotate_qk_fast_bnhd(rotary_embed, q, k):
    cos, sin = cached_rotary_cos_sin(rotary_embed, q.shape[1], q.device, q.dtype)
    return apply_rotary_emb_fast(cos, sin, q), apply_rotary_emb_fast(cos, sin, k)


def qkv_to_bnhd(qkv, heads):
    b, n, _ = qkv.shape
    return qkv.view(b, n, 3, heads, -1).unbind(dim=2)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self._gamma_dtype_cache = {}

    def forward(self, x):
        if not self.training and x.dtype in (torch.float16, torch.bfloat16):
            key = (x.device.type, x.device.index, x.dtype, self.gamma.data_ptr(), self.gamma._version)
            gamma = self._gamma_dtype_cache.get(key)
            if gamma is None:
                gamma = self.gamma.detach().to(device=x.device, dtype=x.dtype)
                self._gamma_dtype_cache.clear()
                self._gamma_dtype_cache[key] = gamma
            return F.rms_norm(x, (x.shape[-1],), gamma, eps=1e-12)
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            shared_qkv_bias=None,
            shared_out_bias=None,
            rotary_embed=None,
            flash=True,
    ):
        super().__init__()
        self.heads = heads
        dim_inner = heads * dim_head
        self.flash = flash
        self.dropout = dropout
        self.rotary_embed = rotary_embed
        self.attend = Attend(flash=flash, dropout=dropout)
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=(shared_qkv_bias is not None))
        if shared_qkv_bias is not None:
            self.to_qkv.bias = shared_qkv_bias

        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=(shared_out_bias is not None)),
            nn.Dropout(dropout)
        )
        if shared_out_bias is not None:
            self.to_out[0].bias = shared_out_bias

    def forward(self, x):
        x = self.norm(x)
        q, k, v = qkv_to_bnhd(self.to_qkv(x), self.heads)

        if self.rotary_embed is not None:
            q, k = rotate_qk_fast_bnhd(self.rotary_embed, q, k)

        if self.flash:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.
            ).transpose(1, 2)
        else:
            out = self.attend(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2)
            ).transpose(1, 2)

        gates = self.to_gates(x)
        out = out * gates.unsqueeze(-1).sigmoid()
        out = out.flatten(start_dim=-2)
        return self.to_out(out)


class Transformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True,
            shared_qkv_bias=None,
            shared_out_bias=None,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    shared_qkv_bias=shared_qkv_bias,
                    shared_out_bias=shared_out_bias,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                ),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
