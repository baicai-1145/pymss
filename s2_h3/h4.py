"""H4 / H4s (Hourglass Band-Rate Allocation): per-layer temporal rate & latent-band allocation derived from the
18-module frozen sensitivity audit (.tmp/msc-band-probe/sensitivity_full12.json).

Core principle: preserve the full-frame (690f) x full-band (62b) residual backbone and the two acoustically critical
front-end modules (L0 freq FFN = complex-STFT phase demodulator; L1 time MSC = fine transient/onset filter), while
allocating frame rate and band rank to each mid/deep residual writer according to its measured information bandwidth:

  L0: time  = StridedLatentPooledKV (q=4, kv=4, r=16)   | freq = BlockStaticBandMixer(s=4)  + FeedForward(s=1)
  L1: time  = full MSC (s=1, k=31, 62b)                 | freq = BlockStaticBandMixer(s=4)  + BlockFeedForward(s=2 / 4)
  L2: time  = StridedLatentMSC (s=2, k=15, r=32)        | freq = BlockStaticBandMixer(s=2/4)+ BlockFeedForward(s=4)
  L3: time  = StridedLatentMSC (s=4, k=9, r=32)         | freq = BlockSlowBandAttention(s=4)+ BlockFeedForward(s=4)
  L4: time  = StridedPooledKV (q=2, kv=4, full 62b)     | freq = BlockSlowBandAttention(s=4)+ BlockFeedForward(s=4)
  L5: time  = StridedLatentMSC (s=2/4, k=15/9, r=32)    | freq = BlockSlowBandAttention(s=2/4)+ BlockFeedForward(s=4)

H4  = balanced variant (L1.ffn s=2, L2.attn s=2, L5.time s=2 k=15, L5.attn s=2)
H4s = 10x speed variant (L1.ffn s=4, L2.attn s=4, L5.time s=4 k=9, L5.attn s=4)
"""
import torch
import torch.nn.functional as F
from torch import nn

from h3 import BANDS, V_WIDTH, BlockFeedForward, StaticBandMixer, block_mean
from pymss_core.modules.bs_roformer.transformer import (
    RMSNorm,
    apply_rotary_emb_fast,
    cached_rotary_cos_sin,
    rotate_qk_fast_bnhd,
)


class StridedLatentMSC(nn.Module):
    """MSC running at block-pooled frame rate `stride` and optional learned `bands_r` latent bands."""
    def __init__(self, dim, stride=2, k=15, bands_r=32, mult=4, dilations=(1,), dropout=0.0, gate_dropout=False):
        super().__init__()
        from entry import MSC
        self.stride, self.k, self.bands_r = stride, k, bands_r
        self.msc = MSC(dim=dim, mult=mult, dilations=dilations, k=k, dropout=dropout, gate_dropout=gate_dropout)
        if bands_r < BANDS:
            q = torch.empty(bands_r, BANDS)
            nn.init.orthogonal_(q)
            self.band_down = nn.Parameter(q.clone())
            self.band_up = nn.Parameter(q.T.clone())
        else:
            self.band_down = None
            self.band_up = None

    def forward(self, x):  # (b, t, band, dim)
        t = x.shape[1]
        z = block_mean(x, self.stride) if self.stride > 1 else x
        if self.band_down is not None:
            z = torch.matmul(self.band_down, z)
        z = self.msc(z)
        if self.band_up is not None:
            z = torch.matmul(self.band_up, z)
        if self.stride > 1:
            z = z.repeat_interleave(self.stride, 1)[:, :t]
        return z


class StridedPooledKVAttention(nn.Module):
    """Temporal attention with query stride `q_stride`, K/V stride `kv_stride`, and optional `bands_r` latent bands."""
    heads = 2
    head_dim = 64

    def __init__(self, dim, rotary, dropout, q_stride=2, kv_stride=4, bands_r=BANDS):
        super().__init__()
        width = self.heads * self.head_dim
        self.q_stride, self.kv_stride, self.bands_r = q_stride, kv_stride, bands_r
        self.query = nn.Linear(dim, width)
        self.key_value = nn.Linear(dim, 2 * width)
        self.gates = nn.Linear(dim, self.heads)
        self.output = nn.Linear(width, dim)
        self.rotary = rotary
        self.dropout = dropout
        if bands_r < BANDS:
            q = torch.empty(bands_r, BANDS)
            nn.init.orthogonal_(q)
            self.band_down = nn.Parameter(q.clone())
            self.band_up = nn.Parameter(q.T.clone())
        else:
            self.band_down = None
            self.band_up = None

    @staticmethod
    def _centers(frames, stride, device):
        starts = torch.arange(0, frames, stride, device=device)
        return starts + ((frames - starts).clamp(max=stride) - 1) // 2

    def forward(self, x):  # (b, t, band, dim)
        b, frames, bands, dim = x.shape
        qs, kvs = self.q_stride, self.kv_stride
        xq = block_mean(x, qs) if qs > 1 else x
        if self.band_down is not None:
            xq = torch.matmul(self.band_down, xq)
        if kvs == qs:
            xkv = xq
        else:
            xkv = block_mean(x, kvs)
            if self.band_down is not None:
                xkv = torch.matmul(self.band_down, xkv)
        r_bands = xq.shape[2]
        tq, tkv = xq.shape[1], xkv.shape[1]
        xq_f = xq.transpose(1, 2).reshape(b * r_bands, tq, dim)
        xkv_f = xkv.transpose(1, 2).reshape(b * r_bands, tkv, dim)

        query = self.query(xq_f).unflatten(-1, (self.heads, self.head_dim))
        key, value = self.key_value(xkv_f).unflatten(-1, (2, self.heads, self.head_dim)).unbind(-3)

        cos, sin = cached_rotary_cos_sin(self.rotary, frames, query.device, query.dtype)
        q_pos = self._centers(frames, qs, x.device)
        kv_pos = q_pos if kvs == qs else self._centers(frames, kvs, x.device)
        query = apply_rotary_emb_fast(cos.index_select(1, q_pos), sin.index_select(1, q_pos), query).transpose(1, 2)
        key = apply_rotary_emb_fast(cos.index_select(1, kv_pos), sin.index_select(1, kv_pos), key).transpose(1, 2)
        value = value.transpose(1, 2).contiguous()

        out = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2) * self.gates(xq_f).sigmoid().unsqueeze(-1)
        out = self.output(out.flatten(-2)).reshape(b, r_bands, tq, dim).transpose(1, 2).contiguous()
        if self.band_up is not None:
            out = torch.matmul(self.band_up, out)
        if qs > 1:
            out = out.repeat_interleave(qs, 1)[:, :frames]
        return out


class BlockStaticBandMixer(nn.Module):
    """StaticBandMixer evaluated once per `stride`-frame block and broadcast back."""
    def __init__(self, dim, heads, stride, dropout=0.0, norm=None):
        super().__init__()
        self.mixer = StaticBandMixer(dim, heads, dropout=dropout, norm=norm)
        self.stride = stride

    def set_cuda_attention_backend(self, backend=None): pass
    def set_mps_attention_backend(self, backend=None, min_tokens=128): pass

    def forward(self, x):  # (b, t, band, dim)
        if self.stride <= 1:
            return self.mixer(x)
        out = self.mixer(block_mean(x, self.stride))
        return out.repeat_interleave(self.stride, 1)[:, :x.shape[1]]


class BlockSlowBandAttention(nn.Module):
    """Full band-axis attention evaluated once per `stride`-frame block (Q/K/V/gates/out all at block rate)."""
    def __init__(self, dim, heads, dim_head, v_width, stride, dropout=0.0, rotary_embed=None, norm=None):
        super().__init__()
        self.heads, self.dim_head, self.v_dim, self.stride = heads, dim_head, v_width // heads, stride
        self.rotary_embed, self.dropout = rotary_embed, dropout
        self.norm = RMSNorm(dim) if norm is None else norm
        self.to_qk = nn.Linear(dim, 2 * heads * dim_head, bias=False)
        self.to_v = nn.Linear(dim, v_width, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(v_width, dim, bias=False), nn.Dropout(dropout))

    def set_cuda_attention_backend(self, backend=None): pass
    def set_mps_attention_backend(self, backend=None, min_tokens=128): pass

    def forward(self, x):  # (b, t, band, dim)
        b, t, f, _ = x.shape
        h, dh, dv, s = self.heads, self.dim_head, self.v_dim, self.stride
        xp = block_mean(x, s) if s > 1 else x
        y = self.norm(xp)
        blocks = y.shape[1]
        bt = b * blocks
        q, k = self.to_qk(y).view(bt, f, 2, h, dh).unbind(2)
        if self.rotary_embed is not None:
            q, k = rotate_qk_fast_bnhd(self.rotary_embed, q, k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = self.to_v(y).view(bt, f, h, dv).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) * dh ** -0.5, -1)
        if self.training and self.dropout:
            a = F.dropout(a, self.dropout)
        # gates must follow the flattened (b*blocks, f, ...) layout of `a @ v`: applied to the 4-D `y` they broadcast as
        # (b, blocks, f, h, 1) against (b*blocks, f, h, dv), which raises for b > 1 and silently mis-gates for b == 1.
        gates = self.to_gates(y.reshape(bt, f, -1)).sigmoid().unsqueeze(-1)
        out = (a @ v).transpose(1, 2) * gates
        out = self.to_out(out.flatten(-2)).view(b, blocks, f, -1)
        return out.repeat_interleave(s, 1)[:, :t] if s > 1 else out


H4_PROFILES = {
    'H4': dict(
        time_cfg=(
            ('pkv', 4, 4, 16),     # L0: r95=4, t_post4=+0.00
            ('msc', 1, 31, 62),    # L1: sacred fine transient MSC
            ('msc', 2, 15, 32),    # L2: t_post2=-0.07, b_svd32=-0.06
            ('msc', 4, 9, 32),     # L3: t_post4=-0.02, b_svd32=-0.01
            ('pkv', 2, 4, 62),     # L4: band-sensitive, t_post2=+0.00
            ('msc', 2, 15, 32),    # L5: t_post2=-0.02, b_svd32=-0.01
        ),
        freq_attn_stride=(4, 4, 2, 4, 4, 2),
        freq_ffn_stride=(1, 2, 4, 4, 4, 4),
        v_width=V_WIDTH,
    ),
    'H4s': dict(
        time_cfg=(
            ('pkv', 4, 4, 16),     # L0
            ('msc', 1, 31, 62),    # L1
            ('msc', 2, 15, 32),    # L2
            ('msc', 4, 9, 32),     # L3
            ('pkv', 2, 4, 62),     # L4
            ('msc', 4, 9, 32),     # L5: t_post4=-0.10, b_svd32=-0.01
        ),
        freq_attn_stride=(4, 4, 4, 4, 4, 4),
        freq_ffn_stride=(1, 4, 4, 4, 4, 4),
        v_width=V_WIDTH,
    ),
}


def apply_h4(model, cfg, arm='H4'):
    """Rewire both time and frequency axes of a built ArmModel into H4 / H4s."""
    prof = H4_PROFILES[arm]
    rotary = model.layers[0][1].layers[0][0].rotary_embed
    with torch.random.fork_rng(devices=[]):
        # Time axis
        for i, spec in enumerate(prof['time_cfg']):
            torch.manual_seed(6400 + i)
            kind = spec[0]
            if kind == 'pkv':
                _, qs, kvs, r = spec
                model.convs[i] = StridedPooledKVAttention(192, rotary, cfg.model.attn_dropout, q_stride=qs, kv_stride=kvs, bands_r=r)
            elif kind == 'msc':
                _, s, k, r = spec
                if s == 1 and k == 31 and r == BANDS:
                    continue  # keep stock full-rate full-band MSC on L1
                model.convs[i] = StridedLatentMSC(192, stride=s, k=k, bands_r=r, mult=4, dilations=(1,), dropout=cfg.model.ff_dropout)
        # Frequency axis
        for i, (_, frequency) in enumerate(model.layers):
            attention, feedforward = frequency.layers[0]
            torch.manual_seed(6450 + i)
            as_ = prof['freq_attn_stride'][i]
            fs_ = prof['freq_ffn_stride'][i]
            if i < 3:
                frequency.layers[0][0] = BlockStaticBandMixer(192, attention.heads, as_, cfg.model.attn_dropout, norm=attention.norm)
            else:
                frequency.layers[0][0] = BlockSlowBandAttention(
                    192, attention.heads, 64, prof['v_width'], as_, cfg.model.attn_dropout, attention.rotary_embed, norm=attention.norm
                )
            if fs_ > 1:
                frequency.layers[0][1] = BlockFeedForward(feedforward, fs_)
    model.freq_4d = True
    return model
