"""H3: the frequency axis of H2 restructured after the frozen-weight audit (.tmp/freq-attn-audit/FINDINGS.md, H).

stacks 0-2  StaticBandMixer    gated, content-independent band routing (frozen oracle `static`: -0.02 .. -0.04 dB per layer):
                               norm -> to_v -> per-head (8 x 62 x 62) band matrix -> gate -> to_out.  No q / k, no RoPE.
            FeedForward       unchanged, every frame (frozen oracle: pooling the FFN of stacks 0-2 costs -11 dB).
stacks 3-5  SlowBandAttention dynamic per-band routing kept; q / k from stride-frame mean tokens, shared by the block
                               (frozen oracle `poolqk4`: -0.15 dB); v / out narrowed to v_width (128); RoPE unchanged.
            BlockFeedForward  FFN on stride-frame mean tokens, broadcast over the block (oracle `ffpool4` on 3-5: -0.31 dB).

Every module takes the (b, t, band, dim) grid.  The time axis (4 MSC + 2 PooledKV) is H2's.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from pymss_core.modules.bs_roformer.transformer import RMSNorm, rotate_qk_fast_bnhd

BANDS, STRIDE, STATIC_STACKS, V_WIDTH = 62, 4, 3, 128


def block_mean(x, stride):
    """Mean over successive stride-frame blocks along dim 1; a trailing partial block repeats its last frame."""
    frames = x.shape[1]
    blocks = -(-frames // stride)
    if blocks * stride != frames:
        x = torch.cat([x, x[:, -1:].expand(x.shape[0], blocks * stride - frames, *x.shape[2:])], 1)
    # sum * reciprocal instead of .mean: .mean lowers to a fp32 ReduceMean+RealDiv pair on Ascend (4.2 + part of
    # the 23.8 ms RealDiv in the H4s profile); the reciprocal is a compile-time constant.
    return x.reshape(x.shape[0], blocks, stride, *x.shape[2:]).sum(2) * (1.0 / stride)



_EXPAND_INDEX: dict = {}


def expand_blocks(out, stride, frames):
    """`out.repeat_interleave(stride, 1)[:, :frames]` as one gather.

    The interleave materialises a `stride`-times larger tensor and the crop then copies the part it keeps -- two full
    passes over up to 4x the data, which showed up as the single largest op in the 219 ms profile (a 3-call, 11.9 ms
    `Slice`).  Every kept frame `i` simply comes from block `i // stride`, so an `index_select` produces the identical
    tensor in one pass and without the 4x intermediate.  Pure gather: bit-exact, so the tier-1 gate applies.
    """
    if stride <= 1:
        return out
    key = (frames, stride, out.device, out.dtype)
    idx = _EXPAND_INDEX.get(key)
    if idx is None:
        idx = (torch.arange(frames, device=out.device) // stride)
        _EXPAND_INDEX.clear()
        _EXPAND_INDEX[key] = idx
    return out.index_select(1, idx)


class StaticBandMixer(nn.Module):
    def __init__(self, dim, heads, dropout=0.0, norm=None):
        super().__init__()
        self.heads, self.norm = heads, RMSNorm(dim) if norm is None else norm
        self.to_v = nn.Linear(dim, dim, bias=False)  # H2's attention projections carry no bias
        # identity + small mixing: every head starts as its own band and learns the rest
        self.weight = nn.Parameter(torch.eye(BANDS).repeat(heads, 1, 1) + torch.randn(heads, BANDS, BANDS) / math.sqrt(2 * BANDS))
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def set_cuda_attention_backend(self, backend=None): pass
    def set_mps_attention_backend(self, backend=None, min_tokens=128): pass

    def forward(self, x):  # (..., band, dim)
        y = self.norm(x)
        h, bands, dim = self.heads, y.shape[-2], y.shape[-1]
        c = dim // h
        lead, flat = y.shape[:-2], 1
        for size in lead:
            flat *= size
        # One batched GEMM per head instead of a broadcast matmul that lowers to `flat` tiny (62, 62) @ (62, c) calls:
        # 8 GEMMs of (62, 62) @ (62, flat * c).  Same arithmetic, a fraction of the launches.
        v = self.to_v(y).reshape(flat, bands, h, c).permute(2, 1, 0, 3).reshape(h, bands, flat * c)
        z = torch.bmm(self.weight, v).reshape(h, bands, flat, c).permute(2, 1, 0, 3)
        if self.training and self.drop.p:
            z = self.drop(z)
        z = z * self.to_gates(y).reshape(flat, bands, h).sigmoid().unsqueeze(-1)
        return self.to_out(z.reshape(*lead, bands, h * c))


class SlowBandAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, v_width, stride, dropout=0.0, rotary_embed=None, norm=None):
        super().__init__()
        self.heads, self.dim_head, self.v_dim, self.stride = heads, dim_head, v_width // heads, stride
        self.rotary_embed, self.dropout = rotary_embed, dropout
        self.norm = RMSNorm(dim) if norm is None else norm
        self.to_qk = nn.Linear(dim, 2 * heads * dim_head, bias=False)  # like H2's to_qkv / to_out
        self.to_v = nn.Linear(dim, v_width, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(v_width, dim, bias=False), nn.Dropout(dropout))

    def set_cuda_attention_backend(self, backend=None): pass
    def set_mps_attention_backend(self, backend=None, min_tokens=128): pass

    def forward(self, x):  # (b, t, band, dim)
        b, t, f, _ = x.shape
        h, dh, dv, s = self.heads, self.dim_head, self.v_dim, self.stride
        y = self.norm(x)
        pooled = self.norm(block_mean(x, s))  # (b, T, band, dim): q / k see the mean token of the block
        q, k = self.to_qk(pooled).view(b, pooled.shape[1], f, 2, h, dh).unbind(3)
        if self.rotary_embed is not None:  # the stock band-axis rotation, batch and block axes merged
            q, k = rotate_qk_fast_bnhd(self.rotary_embed, q.flatten(0, 1), k.flatten(0, 1))
        q, k = q.view(b, -1, f, h, dh).permute(0, 3, 1, 2, 4), k.view(b, -1, f, h, dh).permute(0, 3, 1, 2, 4)  # (b, h, T, band, dh)
        a = torch.softmax(q @ k.transpose(-1, -2) * dh ** -0.5, -1)  # (b, h, T, query band, key band)
        if self.training and self.dropout:
            a = F.dropout(a, self.dropout)
        v = self.to_v(y).view(b, t, f, h, dv).permute(0, 3, 1, 2, 4)  # (b, h, t, band, dv), every frame
        blocks = a.shape[2]
        if blocks * s != t:
            v = torch.cat([v, v[:, :, -1:].expand(-1, -1, blocks * s - t, -1, -1)], 2)
        # The block matmul used to broadcast `a` over the s sub-frames (a.unsqueeze(3) @ v.reshape(...)), which
        # materialised an (s, f, f) copy of the attention matrix: 267 MB per layer at the 8 s shape.  Four bmms
        # reuse the same `a` instead, so nothing is expanded.
        from pymss_core.modules.bs_roformer.transformer import FAST
        if FAST['band_bmm']:
            vb = v.reshape(b, h, blocks, s, f, dv).permute(3, 0, 1, 2, 4, 5).reshape(s, b * h * blocks, f, dv)
            ab = a.reshape(b * h * blocks, f, f)
            out = torch.stack([torch.bmm(ab, vb[i]) for i in range(s)], 0)                 # (s, N, band, dv)
            out = out.reshape(s, b, h, blocks, f, dv).permute(1, 2, 3, 0, 4, 5).flatten(2, 3)[:, :, :t]
        else:
            out = (a.unsqueeze(3) @ v.reshape(b, h, blocks, s, f, dv)).flatten(2, 3)[:, :, :t]
        out = out.permute(0, 2, 3, 1, 4) * self.to_gates(y).sigmoid().unsqueeze(-1)  # (b, t, band, h, dv)
        return self.to_out(out.flatten(-2))


class BlockFeedForward(nn.Module):
    def __init__(self, feedforward, stride):
        super().__init__()
        self.net, self.stride = feedforward.net, stride

    def forward(self, x):  # (b, t, band, dim)
        out = self.net(block_mean(x, self.stride))
        return expand_blocks(out, self.stride, x.shape[1])


def apply_h3(model, cfg, static=STATIC_STACKS, stride=STRIDE, v_width=V_WIDTH):
    """Rewire the frequency transformer of a built H2 layout into H3; new inits stay out of the caller's RNG stream."""
    with torch.random.fork_rng(devices=[]):
        for i, (_, frequency) in enumerate(model.layers):
            attention, feedforward = frequency.layers[0]
            torch.manual_seed(6300 + i)
            if i < static:
                frequency.layers[0][0] = StaticBandMixer(192, attention.heads, cfg.model.attn_dropout, norm=attention.norm)
                continue
            frequency.layers[0][0] = SlowBandAttention(192, attention.heads, 64, v_width, stride, cfg.model.attn_dropout,
                                                       attention.rotary_embed, norm=attention.norm)
            frequency.layers[0][1] = BlockFeedForward(feedforward, stride)
    model.freq_4d = True
    return model
