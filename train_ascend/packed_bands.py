"""Packed alternatives to the per-band Linear groups in BandSplit / MaskEstimator.

The 62 bands carry 7 distinct input widths (4/8/24/48/96/256/258 actually: 4, 8, 12, ...),
and each band owns its own RMSNorm+Linear. ``pymss_core`` ships a grouped path that runs
one bmm per *contiguous equal-width run*; this module adds a single padded bmm
(pad every band to max(width)) so the whole group becomes one kernel.

Both are verified against the per-band path and timed in ``bench_packing.py``.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def _cache_dict(module, name: str) -> dict:
    d = getattr(module, name, None)
    if d is None:
        d = {}
        setattr(module, name, d)
    return d


def packed_band_split_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Padded single-bmm BandSplit: (..., total_in) -> (..., n_bands, dim)."""
    widths = self.dim_inputs
    n_bands = len(widths)
    dim = self.to_features[0][1].out_features
    dev, dt = x.device, x.dtype
    key = (dev.type, dev.index, dt, dim, n_bands)
    cache = _cache_dict(self, "_packed_cache")
    entry = cache.get(key)
    if entry is None:
        pad = max(widths)
        weight = torch.zeros(n_bands, pad, dim, dtype=dt, device=dev)
        bias = torch.zeros(n_bands, dim, dtype=dt, device=dev)
        # the per-band RMSNorm operates on the band's INPUT width, not on `dim`
        gamma = torch.zeros(n_bands, pad, dtype=dt, device=dev)
        for i, (norm, lin) in enumerate(self.to_features):
            w = widths[i]
            weight[i, :w, :] = lin.weight.to(dev, dt).t()  # Linear weight is (dim, w)
            bias[i] = lin.bias.to(dev, dt)
            gamma[i, :w] = norm.gamma.to(dev, dt) * (w ** 0.5)
        # one gather instead of 62 slice-assigns (each assign would keep a full copy alive
        # for the backward and blow the 128 GB of HBM)
        # padded slots point at the appended zero element (index == total input width)
        idx = torch.full((n_bands, pad), sum(widths), dtype=torch.long, device=dev)
        off = 0
        for i, w in enumerate(widths):
            idx[i, :w] = torch.arange(off, off + w, device=dev)
            off += w
        entry = (pad, weight, bias, gamma, idx.reshape(-1))
        cache.clear()
        cache[key] = entry
    pad, weight, bias, gamma, idx = entry

    lead = x.shape[:-1]
    total = x.shape[-1]
    # index `total` (a zero slot appended below) for every padded position
    x_ext = torch.cat([x, x.new_zeros(*x.shape[:-1], 1)], dim=-1)
    xp = x_ext[..., idx].reshape(*lead, n_bands, pad)
    # zero padding does not change ||x|| , so normalising over `pad` equals normalising
    # over the true width (same trick the grouped path uses via slicing).
    xp = F.normalize(xp, dim=-1) * gamma
    flat = xp.reshape(-1, n_bands, pad)
    out = torch.baddbmm(bias.unsqueeze(1), flat.transpose(0, 1), weight)  # (b, pad, dim)
    out = out.transpose(0, 1).reshape(*lead, n_bands, dim)
    return out


def padded_mask_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Padded bmm version of MaskEstimator's per-band MLP heads.

    The per-band heads share the input width but have different *output* widths
    (2 * band width), so the last Linear is padded to 2*max(width) and sliced per band.
    """
    dim_inputs = self.dim_inputs
    n_bands = len(dim_inputs)
    dev, dt = x.device, x.dtype
    dim_in = x.shape[-1]
    key = (dev.type, dev.index, dt, n_bands, dim_in)
    cache = _cache_dict(self, "_packed_mask_cache")
    entry = cache.get(key)
    if entry is None:
        hidden = self.to_freqs[0][0][0].out_features
        pad_out = 2 * max(dim_inputs)
        lin1 = torch.zeros(n_bands, hidden, dim_in, dtype=dt, device=dev)
        b1 = torch.zeros(n_bands, hidden, dtype=dt, device=dev)
        lin2 = torch.zeros(n_bands, pad_out, hidden, dtype=dt, device=dev)
        b2 = torch.zeros(n_bands, pad_out, dtype=dt, device=dev)
        for i, mlp in enumerate(self.to_freqs):
            lin1[i] = mlp[0][0].weight.to(dev, dt)
            b1[i] = mlp[0][0].bias.to(dev, dt)
            w2 = 2 * dim_inputs[i]
            lin2[i, :w2] = mlp[0][2].weight.to(dev, dt)
            b2[i, :w2] = mlp[0][2].bias.to(dev, dt)
        entry = (hidden, lin1, b1, lin2, b2, pad_out)
        cache.clear()
        cache[key] = entry
    hidden, lin1, b1, lin2, b2, pad_out = entry

    lead = x.shape[:-1]
    flat = x.reshape(-1, dim_in)                                  # (N, dim)
    flat_b = flat.transpose(0, 1).unsqueeze(0).expand(n_bands, -1, -1)   # (n_bands, dim, N)
    h = torch.tanh(torch.baddbmm(b1.unsqueeze(-1), lin1, flat_b))        # (n_bands, hidden, N)
    o = torch.baddbmm(b2.unsqueeze(-1), lin2, h)                         # (n_bands, pad_out, N)
    outs = []
    for i, w in enumerate(dim_inputs):
        seg = o[i, :2 * w]                                           # (2w, N)
        a, g = seg.chunk(2, dim=0)
        outs.append((a * torch.sigmoid(g)).transpose(0, 1).reshape(*lead, w))
    return torch.cat(outs, dim=-1)


def enable_packed_band_split(enabled: bool = True) -> None:
    from pymss_core.modules.bs_roformer.bands import BandSplit

    original = getattr(BandSplit, "_per_band_forward", None)
    if enabled:
        if original is None:
            BandSplit._per_band_forward = BandSplit.forward
        BandSplit.forward = packed_band_split_forward
    elif original is not None:
        BandSplit.forward = original
        BandSplit._per_band_forward = None


def enable_padded_mask(enabled: bool = True) -> None:
    from pymss_core.modules.bs_roformer.bands import MaskEstimator

    original = getattr(MaskEstimator, "_per_band_forward", None)
    if enabled:
        if original is None:
            MaskEstimator._per_band_forward = MaskEstimator.forward
        MaskEstimator.forward = padded_mask_forward
    elif original is not None:
        MaskEstimator.forward = original
        MaskEstimator._per_band_forward = None
