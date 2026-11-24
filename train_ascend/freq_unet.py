"""Target-only frequency U-Net for vocal separation (Ascend/NPU training prototype).

Structure (per spec, no teacher weights, no H2 checkpoint):

    band_split(62 bands, dim=192)
      -> enc62   LocalBandBlock(62 bands)
      -> BandDown 62 -> 32 (+pad 1 at the 62-band stage)
      -> enc32   LocalBandBlock
      -> BandDown 32 -> 16
      -> enc16   LocalBandBlock
      -> BandDown 16 -> 8  (channel change -> stack dim)
      -> 4 x ( MSC temporal dilated conv, 8-band full frequency attention w/ RoPE )
      -> BandUp 8 -> 16 + gated skip(enc16) -> dec16 LocalBandBlock
      -> BandUp 16 -> 32 + gated skip(enc32) -> dec32
      -> BandUp 32 -> 62 + gated skip(enc62) -> dec62
      -> final_norm -> MaskEstimationModule -> mask_to_complex_shape

Time axis is never subsampled: every conv that touches time keeps stride 1.

Math blocks are imported from pymss_core so the numerical path (RMSNorm,
Attention+RoPE+gating, BandSplit, MaskEstimationModule, mask_to_complex_shape)
is the same code the CUDA/G4 reference runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList

from pymss_core.modules.bs_roformer.bands import BandSplit, MaskEstimator
from pymss_core.modules.bs_roformer.common import RotaryEmbedding, mask_to_complex_shape
from pymss_core.modules.bs_roformer.transformer import Attention, RMSNorm


# ---------------------------------------------------------------- band layout
def log_band_specs(n_bands: int, n_freq: int) -> list[tuple[int, int]]:
    """Contiguous 62-band split over ``n_freq`` bins, dense in the low end."""
    edges = [0]
    for i in range(1, n_bands):
        edges.append(int(round(n_freq * (i / n_bands) ** 2)))
    edges.append(n_freq)
    # enforce strictly increasing, at least one bin per band
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    edges[-1] = n_freq
    # if we ran out of bins, walk backwards to keep the tail legal
    for i in range(len(edges) - 2, -1, -1):
        if edges[i] >= edges[i + 1]:
            edges[i] = edges[i + 1] - 1
    if edges[0] != 0:
        raise ValueError(f"band split needs >= {n_bands} bins, got {n_freq}")
    return [(edges[i], edges[i + 1]) for i in range(n_bands)]


# ------------------------------------------------------------------- building blocks
class LocalBandBlock(Module):
    """RMSNorm -> depthwise Conv1d(k=7, groups=dim) -> Linear -> +x, then RMSNorm -> SwiGLU -> +x."""

    def __init__(self, dim: int, kernel: int = 7, mult: int = 2):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.p1 = nn.Linear(dim, dim)
        self.n2 = RMSNorm(dim)
        self.gate = nn.Linear(dim, 2 * mult * dim)
        self.p2 = nn.Linear(mult * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F, C) -- depthwise conv runs along the band axis F
        b, t, f, c = x.shape
        h = self.n1(x).reshape(b * t, f, c).transpose(1, 2)
        h = self.dw(h).transpose(1, 2).reshape(b, t, f, c)
        x = x + self.p1(h)
        a, g = self.gate(self.n2(x)).chunk(2, dim=-1)
        return x + self.p2(F.silu(a) * g)


class BandDown(Module):
    """Conv1d(k=4, stride=2, padding=1) along the band axis; pad 1 first for 62 bands."""

    def __init__(self, dim_in: int, dim_out: int, pad_odd: bool = False):
        super().__init__()
        self.pad_odd = pad_odd
        self.conv = nn.Conv1d(dim_in, dim_out, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f, c = x.shape
        h = x.reshape(b * t, f, c).transpose(1, 2)
        if self.pad_odd:
            h = F.pad(h, (1, 1))
        h = self.conv(h)
        return h.transpose(1, 2).reshape(b, t, h.shape[-1], h.shape[1])


class BandUp(Module):
    """ConvTranspose1d(k=4, stride=2, padding=1) + gated skip: up(x) + skip * sigmoid(Linear(skip))."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(dim_in, dim_out, 4, stride=2, padding=1)
        self.gate = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        b, t, f, c = x.shape
        h = self.up(x.reshape(b * t, f, c).transpose(1, 2))
        h = h.transpose(1, 2).reshape(b, t, h.shape[-1], h.shape[1])[:, :, : skip.shape[2]]
        return h + skip * torch.sigmoid(self.gate(skip))


class MSC(Module):
    """Multi-scale (dilated, depthwise) temporal conv with a GLU gate. Time stride is 1."""

    def __init__(self, dim: int, kernel: int = 5, dilations: tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.pw = nn.Linear(dim, 2 * dim)
        self.convs = ModuleList(
            [nn.Conv1d(dim, dim, kernel, padding=d * (kernel - 1) // 2, dilation=d, groups=dim) for d in dilations]
        )
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T, C) at the 8-band stage; conv over T
        b, f, t, c = x.shape
        a, g = self.pw(self.norm(x)).chunk(2, dim=-1)
        h = a.reshape(b * f, t, c).transpose(1, 2)
        mixed = sum(conv(h) for conv in self.convs)
        a = mixed.transpose(1, 2).reshape(b, f, t, c)
        return self.out(a * torch.sigmoid(g))


class FreqAttentionBlock(Module):
    """Full attention across the (8) bands for every time frame, RoPE, heads x dim_head."""

    def __init__(self, dim: int, heads: int = 2, dim_head: int = 64):
        super().__init__()
        if heads * dim_head != dim:
            raise ValueError(f"heads*dim_head must equal dim: {heads}*{dim_head} != {dim}")
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, rotary_embed=RotaryEmbedding(dim_head))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T, C) -> sequence over F, batch over (B*T)
        b, f, t, c = x.shape
        h = x.permute(0, 2, 1, 3).reshape(b * t, f, c)
        h = self.attn(h) + h
        return h.reshape(b, t, f, c).permute(0, 2, 1, 3)


class TemporalResidualBlock(Module):
    """MSC (time) + frequency attention, both residual."""

    def __init__(self, dim: int, heads: int = 2, dim_head: int = 64):
        super().__init__()
        self.msc = MSC(dim)
        self.freq = FreqAttentionBlock(dim, heads=heads, dim_head=dim_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.msc(x)
        return self.freq(x)


# ---------------------------------------------------------------------- config
@dataclass
class FreqUNetConfig:
    n_bands: int = 62
    n_fft: int = 2048  # -> 1025 bins
    band_dim: int = 192  # band_split output dim (spec)
    # enc62/enc32/enc16/8-band-stack. widths chosen so the total lands just under
    # the reference 20,961,844 (see sweep_params.py); 8-band stack is pinned to
    # heads*dim_head = 2*64 = 128 by the spec.
    dims: tuple[int, int, int, int] = (192, 288, 256, 128)
    mult: int = 2
    heads: int = 2
    dim_head: int = 64
    n_temporal_blocks: int = 4
    mlp_dim: int = 384
    mask_depth: int = 2
    mlp_expansion_factor: int = 4
    audio_channels: int = 2
    instruments: tuple[str, ...] = ("vocals",)
    band_specs: list[tuple[int, int]] = field(default_factory=list)

    def resolved_band_specs(self) -> list[tuple[int, int]]:
        if self.band_specs:
            return self.band_specs
        return log_band_specs(self.n_bands, self.n_fft // 2 + 1)


# ------------------------------------------------------------------------ model
class BandUNetSeparator(Module):
    """band_split -> frequency U-Net -> final_norm -> MaskEstimator -> complex mask."""

    def __init__(self, cfg: FreqUNetConfig):
        super().__init__()
        self.cfg = cfg
        specs = cfg.resolved_band_specs()
        widths = tuple(end - start for start, end in specs)
        # pymss_core/BS-RoFormer convention: one band carries 2 (re/im) * channels values
        dim_inputs = tuple(2 * cfg.audio_channels * w for w in widths)
        d62, d32, d16, ds = cfg.dims

        self.band_split = BandSplit(dim=cfg.band_dim, dim_inputs=dim_inputs)
        self.enc62 = LocalBandBlock(d62, mult=cfg.mult)
        self.down62 = BandDown(d62, d32, pad_odd=len(specs) % 2 == 0)  # 62 -> pad -> 64 -> 32
        self.enc32 = LocalBandBlock(d32, mult=cfg.mult)
        self.down32 = BandDown(d32, d16)
        self.enc16 = LocalBandBlock(d16, mult=cfg.mult)
        self.down16 = BandDown(d16, ds)
        self.blocks = ModuleList([TemporalResidualBlock(ds, cfg.heads, cfg.dim_head) for _ in range(cfg.n_temporal_blocks)])
        self.up8 = BandUp(ds, d16)
        self.dec16 = LocalBandBlock(d16, mult=cfg.mult)
        self.up16 = BandUp(d16, d32)
        self.dec32 = LocalBandBlock(d32, mult=cfg.mult)
        self.up32 = BandUp(d32, d62)
        self.dec62 = LocalBandBlock(d62, mult=cfg.mult)
        self.final_norm = RMSNorm(d62)
        self._estimate_masks = MaskEstimator(
            dim=d62,
            dim_inputs=dim_inputs,
            depth=cfg.mask_depth,
            mlp_expansion_factor=cfg.mlp_expansion_factor,
        )

    def forward(self, stft_repr: torch.Tensor) -> torch.Tensor:
        """stft_repr: (B, F*C, T, 2) real view of the STFT (f-major) -> mask (B, F*C, T, 2)."""
        b, fs, t, complex_dim = stft_repr.shape
        features = stft_repr.permute(0, 2, 1, 3).reshape(b, t, fs * complex_dim)
        weights = next(self.band_split.parameters())
        x = self.band_split(features.to(dtype=weights.dtype))

        x = self.enc62(x)
        s62 = x
        x = self.enc32(self.down62(x))
        s32 = x
        x = self.enc16(self.down32(x))
        s16 = x
        x = self.down16(x)  # (B, T, 8, ds)
        x = x.permute(0, 2, 1, 3)  # (B, F, T, C)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1, 3)  # (B, T, F, C)

        x = self.dec16(self.up8(x, s16))
        x = self.dec32(self.up16(x, s32))
        x = self.dec62(self.up32(x, s62))
        mask = self._estimate_masks(self.final_norm(x))
        if mask.dtype != torch.float32:
            mask = mask.float()
        # MaskEstimator returns (B, T, sum(dim_inputs)); add the stem axis back so
        # mask_to_complex_shape sees the (B, N, T, FC) layout the CUDA path uses.
        return mask_to_complex_shape(mask.unsqueeze(1), complex_dim=2)[:, 0]  # (B, F*C, T, 2)


def build_model(cfg: FreqUNetConfig | None = None) -> BandUNetSeparator:
    return BandUNetSeparator(cfg or FreqUNetConfig())


def param_count(module: Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def param_breakdown(module: Module) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name, child in module.named_children():
        sizes[name] = param_count(child)
    return sizes


if __name__ == "__main__":
    cfg = FreqUNetConfig()
    model = build_model(cfg)
    total = param_count(model)
    print(f"bands={cfg.n_bands} n_fft={cfg.n_fft} dims={cfg.dims} total params={total} ({total / 1e6:.2f}M)")
    for name, size in param_breakdown(model).items():
        print(f"  {name:14s} {size:>10,d}")
    specs = cfg.resolved_band_specs()
    print(f"  band widths: min={min(e - s for s, e in specs)} max={max(e - s for s, e in specs)}")
