"""Band U-Net v3: same budget as v1, but the parameters go to cross-band attention at
high frequency resolution instead of depthwise convs that only see +-3 bands.

Layout (per spec, width 192 everywhere, time never subsampled):

    band_split(62 bands, dim=192)                              [pymss_core, unchanged]
      -> BandDown 62 -> 31 : BandBlock(shift=1) + avgpool(1,2) + Linear(192,192)
      -> BandDown 31 -> 16 : same
      -> BandDown 16 ->  8 : same
      -> bottleneck : x2 [RMSNorm -> full attention over the 8 bands (heads=3, dim_head=64,
                       RoPE) -> +x -> RMSNorm -> MLP(4x) -> +x] + 1 MSC(mult=2, k=31,
                       dilations=(1,4,16)) over time
      -> BandUp  8 -> 16 / 16 -> 31 / 31 -> 62 : Linear(192,192) -> repeat_interleave(2)
                     -> crop to skip bands -> x + skip*sigmoid(Linear(skip))
      -> final_norm -> MaskEstimator                             [pymss_core, unchanged]

BandBlock = RMSNorm -> window attention (window=8 bands, learned relative position bias
[heads, 2*window] indexed by (i-j) mod 2*window) -> Linear residual, then
RMSNorm -> SwiGLU(mult=2) -> Linear residual.

Skips are pushed *before* each down (pitfall 3), the pool kernel is (1,2) on the band axis
after permuting to (B, C, T, F) (pitfall 2), and the band-dim pad tuple for a 4-D tensor is
(0, 0, 0, pad) (pitfall 1).

Parameter budget, reconciled against the NAS reference total 19,875,890:
    band_split            803,204   (pymss_core, identical to v1)
    _estimate_masks    15,495,688   (pymss_core, identical to v1)
    final_norm                192
    6 x BandBlock       2,224,736   (370,752 + 16*heads each; heads 2/2/3 enc, 3/2/2 dec)
    3 x down Linear       111,168
    6 x up Linear         222,336
    bottleneck            888,774   (2 x [Attention(heads=3, dim_head=64, no qkv/out bias,
                                     to_gates) + FeedForward(4x, biassed)] + output norm)
    MSC                   129,792
    ---------------------------------
    total              19,875,890
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList

from pymss_core.modules.bs_roformer.bands import BandSplit, MaskEstimator
from pymss_core.modules.bs_roformer.common import RotaryEmbedding, mask_to_complex_shape
from pymss_core.modules.bs_roformer.transformer import RMSNorm, Transformer

from freq_unet import MSC, _TRITON_SILU, log_band_specs, rms_norm

# F.scaled_dot_product_attention with a broadcast additive mask costs 42.05 ms fwd+bwd at the
# 62-band stage (bs24) on this CANN stack; the explicit softmax form costs 22.52 ms for the
# same math (measured, same shapes, bf16). Default to the explicit one.
_ATTN_IMPL = os.environ.get("PYMSS_V3_ATTN", "manual").lower()

# --------------------------------------------------------------------- blocks


class WindowAttention(Module):
    """Attention inside groups of `window` bands, with a learned relative position bias.

    The band axis is padded up to a multiple of `window` (masked out by construction, since
    the padded slots never appear in the output), optionally rolled by half a window first so
    that neighbouring levels group different bands together.
    """

    def __init__(self, dim: int, heads: int, window: int = 8, shift: int = 0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads, self.window, self.shift = heads, window, shift
        self.dim_head = dim // heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.to_out = nn.Linear(dim, dim)
        # [heads, 2*window], indexed by (i - j) mod 2*window
        self.rel_pos = nn.Parameter(torch.zeros(heads, 2 * window))
        idx = (torch.arange(window)[:, None] - torch.arange(window)[None, :]) % (2 * window)
        self.register_buffer("rel_idx", idx, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F, C) -> (B, T, F, C)."""
        b, t, f, c = x.shape
        w = self.window
        h = rms_norm(x, self.norm.gamma)
        if self.shift:
            h = torch.roll(h, -w // 2, dims=2)
        pad = (-f) % w
        if pad:
            # 4-D pad tuple counts from the last dim: (C_left, C_right, F_left, F_right)
            h = F.pad(h, (0, 0, 0, pad))
        fw = f + pad
        qkv = self.to_qkv(h).reshape(b, t, fw // w, w, 3, self.heads, self.dim_head)
        q, k, v = qkv.permute(4, 0, 1, 2, 5, 3, 6).unbind(0)  # (B,T,G,heads,w,dh)
        bias = self.rel_pos[:, self.rel_idx]                     # (heads, w, w)
        if _ATTN_IMPL == "sdpa":
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                scale=self.dim_head ** -0.5)
        else:
            scale = self.dim_head ** -0.5
            logits = torch.einsum("...id,...jd->...ij", q, k) * scale + bias
            out = torch.einsum("...ij,...jd->...id", logits.softmax(dim=-1), v)
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(b, t, fw, c)
        if pad:
            out = out[:, :, :f, :]
        if self.shift:
            out = torch.roll(out, w // 2, dims=2)
        return out          # projection + residual are fused by BandBlock


def _linear_residual(x: torch.Tensor, h: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    """linear(h) + x in one pass.

    Same values as ``x + linear(h)``, but the residual is folded into the GEMM (beta=1) so the
    (B, T, 62, 192) tensor is not read and written twice - the trick that was worth 57 ms/step
    on v1. Weights are cast to the activation dtype because autocast does not reach an
    explicit addmm call.
    """
    shape = x.shape
    flat_h = h.reshape(-1, linear.in_features)
    flat_x = x.reshape(-1, linear.out_features)
    bias = None if linear.bias is None else linear.bias.to(flat_h.dtype)
    out = torch.addmm(bias, flat_h, linear.weight.to(flat_h.dtype).t())
    return out.add_(flat_x.to(out.dtype)).reshape(shape)


class BandBlock(Module):
    """RMSNorm -> window attention -> residual, RMSNorm -> SwiGLU(mult) -> residual."""

    def __init__(self, dim: int, heads: int, window: int = 8, shift: int = 0, mult: int = 2):
        super().__init__()
        self.attn = WindowAttention(dim, heads, window=window, shift=shift)
        self.n2 = RMSNorm(dim)
        self.gate = nn.Linear(dim, 2 * mult * dim)
        self.p = nn.Linear(mult * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _linear_residual(x, self.attn(x), self.attn.to_out)
        z = rms_norm(x, self.n2.gamma)
        a, g = self.gate(z).chunk(2, dim=-1)
        if _TRITON_SILU and a.is_contiguous() and g.is_contiguous():
            from triton_blocks import fused_silu_mul

            act = fused_silu_mul(a, g)   # autograd-safe since the overnight fix
        else:
            act = F.silu(a) * g
        return _linear_residual(x, act, self.p)


class BandDown(Module):
    """BandBlock(shift=1) -> average-pool the band axis by 2 -> Linear."""

    def __init__(self, dim: int, heads: int, window: int = 8):
        super().__init__()
        self.block = BandBlock(dim, heads, window=window, shift=1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)                                  # (B, T, F, C)
        b, t, f, c = x.shape
        if f % 2:
            # pad one band at the end; on (B, T, F, C) the tuple pads C=0, F=(0,1)
            x = F.pad(x, (0, 0, 0, 1))
            f += 1
        # (B, C, T, F) so kernel (1, 2) pools the *band* axis, never the frame axis
        y = x.permute(0, 3, 1, 2)
        y = F.avg_pool2d(y, kernel_size=(1, 2))
        y = y.permute(0, 2, 3, 1)                          # back to (B, T, F/2, C)
        return self.proj(y)


class BandUp(Module):
    """Linear -> repeat_interleave(2) on bands -> crop to skip -> x + skip * sigmoid(Linear(skip))."""

    def __init__(self, dim: int, heads: int, window: int = 8):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.block = BandBlock(dim, heads, window=window, shift=0)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        y = self.proj(x).repeat_interleave(2, dim=2)
        y = y[:, :, : skip.shape[2], :]
        y = y + skip * torch.sigmoid(self.gate(skip))
        return self.block(y)


# ---------------------------------------------------------------------- config
@dataclass
class FreqUNetV3Config:
    n_bands: int = 62
    n_fft: int = 2048
    band_dim: int = 192
    window: int = 8
    levels: tuple[int, int, int, int] = (62, 31, 16, 8)
    heads: tuple[int, int, int, int] = (2, 2, 3, 3)   # 62 / 31 / 16 / 8-band level
    mult: int = 2
    bottleneck_depth: int = 2
    bottleneck_heads: int = 3
    bottleneck_dim_head: int = 64
    bottleneck_ff_mult: int = 4
    msc_kernel: int = 31
    msc_dilations: tuple[int, ...] = (1, 4, 16)
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
class BandUNetV3(Module):
    def __init__(self, cfg: FreqUNetV3Config):
        super().__init__()
        self.cfg = cfg
        specs = cfg.resolved_band_specs()
        widths = tuple(end - start for start, end in specs)
        dim_inputs = tuple(2 * cfg.audio_channels * w for w in widths)
        d = cfg.band_dim
        h62, h31, h16, h8 = cfg.heads
        w = cfg.window

        self.band_split = BandSplit(dim=d, dim_inputs=dim_inputs)
        self.down62 = BandDown(d, h62, window=w)   # 62 -> 31
        self.down31 = BandDown(d, h31, window=w)   # 31 -> 16
        self.down16 = BandDown(d, h16, window=w)   # 16 -> 8
        self.bottleneck = Transformer(
            dim=d, depth=cfg.bottleneck_depth, heads=cfg.bottleneck_heads,
            dim_head=cfg.bottleneck_dim_head, rotary_embed=RotaryEmbedding(cfg.bottleneck_dim_head),
            ff_mult=cfg.bottleneck_ff_mult, norm_output=True,
        )
        self.msc = MSC(d, kernel=cfg.msc_kernel, dilations=cfg.msc_dilations)
        self.up8 = BandUp(d, h16, window=w)        # 8 -> 16
        self.up16 = BandUp(d, h31, window=w)       # 16 -> 31
        self.up31 = BandUp(d, h62, window=w)       # 31 -> 62
        self.final_norm = RMSNorm(d)
        self._estimate_masks = MaskEstimator(
            dim=d, dim_inputs=dim_inputs, depth=cfg.mask_depth,
            mlp_expansion_factor=cfg.mlp_expansion_factor,
        )

    def forward(self, stft_repr: torch.Tensor) -> torch.Tensor:
        """stft_repr: (B, F*C, T, 2) real view -> mask (B, F*C, T, 2)."""
        b, fs, t, complex_dim = stft_repr.shape
        features = stft_repr.permute(0, 2, 1, 3).reshape(b, t, fs * complex_dim)
        weights = next(self.band_split.parameters())
        x = self.band_split(features.to(dtype=weights.dtype))

        skips = []
        skips.append(x)                 # 62 bands -- pushed *before* the down
        x = self.down62(x)
        skips.append(x)                 # 31
        x = self.down31(x)
        skips.append(x)                 # 16
        x = self.down16(x)              # 8

        # bottleneck: frequency attention over the 8 bands per frame, then time conv
        bb, tt, ff, cc = x.shape
        h = x.permute(0, 1, 2, 3).reshape(bb * tt, ff, cc)   # batch = B*T, seq = 8 bands
        h = self.bottleneck(h)
        x = h.reshape(bb, tt, ff, cc)
        x = self.msc(x) + x

        x = self.up8(x, skips.pop())    # -> 16
        x = self.up16(x, skips.pop())   # -> 31
        x = self.up31(x, skips.pop())   # -> 62
        mask = self._estimate_masks(self.final_norm(x))
        if mask.dtype != torch.float32:
            mask = mask.float()
        return mask_to_complex_shape(mask.unsqueeze(1), complex_dim=2)[:, 0]


def build_model(cfg: FreqUNetV3Config | None = None) -> BandUNetV3:
    model = BandUNetV3(cfg or FreqUNetV3Config())
    # same device preference as v1: the per-band path is faster on this CANN stack
    for mod in model.modules():
        if hasattr(mod, "use_grouped_forward"):
            mod.use_grouped_forward = False
    return model


def param_count(module: Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def param_breakdown(module: Module) -> dict[str, int]:
    return {name: param_count(child) for name, child in module.named_children()}


# ------------------------------------------------------------------- self-tests
def _test_window_roundtrip(cfg: FreqUNetV3Config) -> None:
    """Splitting into windows and merging back must be bit-identical to the input."""
    for bands in (62, 31, 16, 8):
        for shift in (0, 1):
            attn = WindowAttention(cfg.band_dim, heads=2, window=cfg.window, shift=shift).eval()
            x = torch.randn(2, 3, bands, cfg.band_dim)
            with torch.no_grad():
                # identity check on the split/merge machinery only
                h = x
                if shift:
                    h = torch.roll(h, -cfg.window // 2, dims=2)
                pad = (-bands) % cfg.window
                if pad:
                    h = F.pad(h, (0, 0, 0, pad))
                fw = bands + pad
                view = h.reshape(2, 3, fw // cfg.window, cfg.window, cfg.band_dim)
                merged = view.reshape(2, 3, fw, cfg.band_dim)
                if pad:
                    merged = merged[:, :, :bands, :]
                if shift:
                    merged = torch.roll(merged, cfg.window // 2, dims=2)
                assert torch.equal(merged, x), f"window roundtrip broke at bands={bands} shift={shift}"
                assert attn.rel_idx.max() < 2 * cfg.window
    print("  window roundtrip: exact (62/31/16/8 bands, shift 0/1)")


def _test_shapes(cfg: FreqUNetV3Config, frames: int = 690, batch: int = 2) -> None:
    model = build_model(cfg)
    model.eval()
    n_freq = cfg.n_fft // 2 + 1
    stft = torch.randn(batch, n_freq * cfg.audio_channels, frames, 2)
    captured = {}
    handles = []

    def hook(name):
        def fn(_mod, _inp, out):
            if torch.is_tensor(out) and out.dim() == 4:
                captured.setdefault(name, tuple(out.shape))
        return fn

    for name in ("down62", "down31", "down16", "up8", "up16", "up31"):
        handles.append(getattr(model, name).register_forward_hook(hook(name)))
    with torch.no_grad():
        mask = model(stft)
    for h in handles:
        h.remove()
    assert tuple(mask.shape) == (batch, n_freq * cfg.audio_channels, frames, 2), mask.shape
    expect = {"down62": 31, "down31": 16, "down16": 8, "up8": 16, "up16": 31, "up31": 62}
    for name, bands in expect.items():
        shape = captured[name]
        assert shape[2] == bands, f"{name} produced {shape}, expected {bands} bands"
        assert shape[3] == cfg.band_dim, f"{name} width {shape[3]} != {cfg.band_dim}"
        assert shape[1] == frames, f"{name} frames {shape[1]} != {frames} (time must not shrink)"
    print(f"  shapes: bands 62/31/16/8 and width {cfg.band_dim} at {frames} frames, "
          f"time preserved, mask {tuple(mask.shape)}")


def _test_backward(cfg: FreqUNetV3Config, frames: int = 690, batch: int = 1) -> None:
    model = build_model(cfg)
    model.train()
    n_freq = cfg.n_fft // 2 + 1
    stft = torch.randn(batch, n_freq * cfg.audio_channels, frames, 2)
    mask = model(stft)
    mask.pow(2).mean().backward()
    total = 0
    grads = 0
    missing = []
    # pymss_core's RotaryEmbedding stores its frequency table as a Parameter; it is a
    # constant, so "no grad" there is expected (v1 behaves the same way).
    constant = ("rotary_embed.freqs",)
    for name, p in model.named_parameters():
        total += 1
        if p.grad is None:
            if not any(k in name for k in constant):
                missing.append(name)
        elif float(p.grad.abs().max()) > 0:
            grads += 1
    assert not missing, f"{len(missing)} params without grad, e.g. {missing[:5]}"
    print(f"  backward: {grads}/{total} params got a non-zero grad, none missing")


def main() -> int:
    cfg = FreqUNetV3Config()
    print("freq_unet_v3 self-check")
    _test_window_roundtrip(cfg)
    _test_shapes(cfg)
    _test_backward(cfg)
    model = build_model(cfg)
    total = param_count(model)
    print(f"  params: {total:,d} (target 19,875,890)")
    for k, v in param_breakdown(model).items():
        print(f"    {k:18s} {v:>10,d}")
    enc = sum(param_count(getattr(model, n)) for n in ("down62", "down31", "down16"))
    print(f"    {'encoder total':18s} {enc:>10,d} (NAS 1.22M)")
    print(f"    {'bottleneck':18s} {param_count(model.bottleneck):>10,d} (NAS 0.89M)")
    print(f"    {'temporal(msc)':18s} {param_count(model.msc):>10,d} (NAS 0.13M)")
    dec = sum(param_count(getattr(model, n)) for n in ("up8", "up16", "up31"))
    print(f"    {'decoder total':18s} {dec:>10,d} (NAS 1.11M + ups)")
    return 0 if total == 19_875_890 else 1


if __name__ == "__main__":
    raise SystemExit(main())
