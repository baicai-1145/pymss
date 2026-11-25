"""Round-2 architecture variants: token-count and block-count reduction.

Motivation (measured, not assumed): at ``dim=192, bands=60, T=800`` the stock model
runs at ~4% of the L4's fp16 peak -- it is kernel-count bound, not FLOP bound.
Every operator swap tried so far (band-axis conv, dilated conv) tied or lost
because it kept the number of kernels.  What won (dualtier) did so by doing
*fewer, larger* ops.  Everything here therefore attacks a multiplicative factor
of

    MACs ~ L * B * T * 24 d^2   (+ 2 * B * T * d  attention)
             ^   ^   ^     ^
             |   |   |     +-- ``bandcount``: fewer bands, wider d, equal FLOPs
             |   |   +-------- (round 3)
             |   +------------ ``bandcount``
             +---------------- ``singleffn``: 24 d^2 -> 12 d^2 per block

``flopmatch`` is the control arm: the stock model narrowed to the same MACs as a
reduced arm, so "the architecture is better" stays separable from "the model is
smaller".

Nothing here touches the STFT, the band-split contract, the mask estimator or
iSTFT, so results stay comparable with every arm measured so far.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList

from .bs_roformer import BSRoformer
from .common import DEFAULT_FREQS_PER_BANDS, RotaryEmbedding, mask_to_complex_shape
from .transformer import Attention, FeedForward

__all__ = ["geometric_bands", "erb_bands", "AxialBlock", "AxialModel", "build_spec", "SPEC_NAMES"]

SPEC_NAMES = ("baseline", "singleffn", "pyramid", "temporal", "convffn", "tailtime")

# (dim, n_blocks) per pyramid stage.  Band counts are derived from the model's
# own band count (default 62) by halving at each stage boundary -- hardcoding 60
# here is exactly the bug that produced a 31-band intermediate.
DEFAULT_PYRAMID = ((192, 2), (224, 2))
# (local_blocks at full frame rate, global_blocks at pooled rate, pool factor)
DEFAULT_TIER = (3, 3, 4)


def erb_bands(num_bands: int, total_bins: int = 1025, min_bins: int = 4, sample_rate: int = 44100) -> tuple[int, ...]:
    """Equal-ERB band widths (low frequencies kept fine), clipped to a bin floor.

    Unlike ``geometric_bands`` (whose widths grow from DC and put a ~323 Hz
    first band exactly on the vocal F0 region), this allocates resolution the
    way hearing does: ~86 Hz bands at the bottom, wide bands above 10 kHz.
    """
    import math

    def erbs_of(f):
        return 21.4 * math.log10(0.00437 * f + 1)

    nyq = sample_rate / 2
    total = erbs_of(nyq)
    edges = [0.0]
    for i in range(1, num_bands):
        e = total * i / num_bands
        edges.append((10 ** (e / 21.4) - 1) / 0.00437)
    edges.append(nyq)
    bins = [max(0, round(f * 2048 / sample_rate)) for f in edges]
    fixed = [bins[0]]
    for bval in bins[1:-1]:
        if bval < fixed[-1] + min_bins:
            bval = fixed[-1] + min_bins
        fixed.append(bval)
    fixed.append(total_bins)
    widths = [fixed[i + 1] - fixed[i] for i in range(len(fixed) - 1)]
    return tuple(widths)


def geometric_bands(num_bands: int, total_bins: int = 1025, min_width: int = 2) -> tuple[int, ...]:
    """Band widths growing geometrically, summing exactly to ``total_bins``.

    Deliberately dumb (one free parameter, no learned boundaries) so that a
    band-count comparison only changes the count.
    """
    if num_bands < 1:
        raise ValueError("num_bands must be >= 1")
    lo, hi = 1.0, 12.0
    best = None
    for _ in range(200):
        ratio = 0.5 * (lo + hi)
        raw = [ratio**i for i in range(num_bands)]
        s = sum(raw)
        widths = [max(min_width, int(round(w / s * total_bins))) for w in raw]
        residual = total_bins - sum(widths)
        widths[widths.index(max(widths))] += residual
        if residual == 0 and min(widths) >= 1:
            return tuple(widths)
        best = widths
        lo, hi = (ratio, hi) if sum(widths) < total_bins else (lo, ratio)
    raise RuntimeError(f"could not build {num_bands} bands summing to {total_bins}: {best} sum={sum(best)}")


class AxialBlock(Module):
    """One axial block over a ``(b, t, f, d)`` grid.

    The stock design spends **two** full-width FFNs per block (one after each
    axis); ``ffn_freq=None`` turns that into the single FFN a standard
    transformer block spends.
    """

    def __init__(self, time_attn: Module, freq_attn: Module, ffn_time: Module, ffn_freq: Module | None):
        super().__init__()
        self.time_attn = time_attn
        self.freq_attn = freq_attn
        self.ffn_time = ffn_time
        self.ffn_freq = ffn_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f, d = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
        x = self.time_attn(seq).reshape(b, f, t, d).permute(0, 2, 1, 3) + x
        if self.ffn_time is not None:
            x = self.ffn_time(x) + x
        x = self.freq_attn(x.reshape(b * t, f, d)).reshape(b, t, f, d) + x
        if self.ffn_freq is not None:
            x = self.ffn_freq(x) + x
        return x


class _AxisStack(Module):
    """``depth`` x (pre-norm Attention + residual) over a 3-D ``(n, seq, d)``."""

    def __init__(self, depth, dim, heads, dim_head, dropout, flash, dim_rotary):
        super().__init__()
        self.layers = ModuleList(
            Attention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                rotary_embed=RotaryEmbedding(dim=dim_rotary),
                flash=flash,
            )
            for _ in range(depth)
        )

    def forward(self, x):
        for attn in self.layers:
            x = attn(x) + x
        return x


class _FFNStack(Module):
    def __init__(self, depth, dim, mult, dropout):
        super().__init__()
        self.layers = ModuleList(FeedForward(dim=dim, mult=mult, dropout=dropout) for _ in range(depth))

    def forward(self, x):
        for ff in self.layers:
            x = ff(x) + x
        return x


class ConvFFN(Module):
    """Gated FFN whose mid-layer is a depthwise *temporal* conv (k=7).

    For the freq-only speed arm this is the only place time context enters:
    Linear(d -> 2h) -> GLU -> DWConv1d over time -> Linear(h -> d).  Cost is
    O(T * d * k) -- ~114x cheaper than the O(T^2 d) time attention it replaces.
    """

    def __init__(self, dim, mult=4, k=7, dropout=0.0, gated=True):
        super().__init__()
        h = dim * mult // 2
        self.gated = bool(gated)
        # gated: GLU needs u,v (2h).  ungated: single h + SiLU -- halves the
        # widest activation, the tensor that dominates inference bandwidth.
        self.proj_in = nn.Linear(dim, h * 2 if self.gated else h)
        self.conv = nn.Conv1d(h, h, k, padding=k // 2, groups=h)
        self.proj_out = nn.Linear(h, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, f, d = x.shape
        p = self.proj_in(x)
        if self.gated:
            u, v = p.chunk(2, dim=-1)
            h = u * torch.sigmoid(v)               # (b, t, f, hid)
        else:
            h = p * torch.sigmoid(p)               # SiLU
        hid = h.shape[-1]
        h = h.permute(0, 2, 1, 3).reshape(b * f, t, hid).transpose(1, 2)  # (b*f, hid, t)
        h = self.conv(h)
        h = h.transpose(1, 2).reshape(b, f, t, hid).permute(0, 2, 1, 3)   # (b, t, f, hid)
        return self.drop(self.proj_out(h))


class AxialModel(BSRoformer):
    """``BSRoformer`` with a flat list of :class:`AxialBlock` instead of pairs.

    The stock STFT / band split / mask estimator / iSTFT path is untouched.
    """

    def _forward_mask_core(self, stft_repr: torch.Tensor) -> torch.Tensor:
        b, fs, model_t, complex_dim = stft_repr.shape
        features = stft_repr.permute(0, 2, 1, 3).reshape(b, model_t, fs * complex_dim)
        weights = next(self.band_split.parameters(), None)
        if weights is not None:
            features = features.to(dtype=weights.dtype)
        x = self.band_split(features)
        return mask_to_complex_shape(self._estimate_masks(self.final_norm(self._stack_forward(x))), complex_dim=2)

    def _stack_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the axial body over ``(b, t, f, d)``."""
        for block in self.axial:
            x = block(x)
        return x


def build_spec(spec: str, *, dim: int, bands=None, layout=None, tier=None, single_ffn=None, conv_k=7,
               conv_gated=True, ffn_every=1, **cfg) -> AxialModel:
    """Build a round-2/3 arm.

    ``spec``:
      ``baseline``   stock axial blocks (two FFNs per block)
      ``singleffn``  one FFN per block
      ``pyramid``    band-axis pyramid: merge bands with depth (see ``layout``)
      ``temporal``   time-axis dual tier: full-rate local + pooled global branch

    ``bands`` overrides ``freqs_per_bands``; ``dim``/``heads``/``dim_head`` are
    supplied by the caller so a FLOP-matched control is expressible.
    """
    spec = str(spec).lower()
    if spec not in SPEC_NAMES:
        raise ValueError(f"unknown spec {spec!r}; expected {SPEC_NAMES}")
    if spec in ("pyramid", "temporal"):
        return _build_compound(spec, dim=dim, bands=bands, layout=layout, tier=tier,
                               single_ffn=bool(single_ffn), cfg=cfg)

    kwargs = dict(cfg)
    if bands is not None:
        kwargs["freqs_per_bands"] = tuple(bands)
    heads = int(kwargs.pop("heads", 8))
    dim_head = int(kwargs.pop("dim_head", 64))
    depth = int(kwargs.pop("depth", 12))
    attn_dropout = float(kwargs.pop("attn_dropout", 0.0))
    ff_dropout = float(kwargs.pop("ff_dropout", 0.0))
    ff_mult = int(kwargs.pop("ff_mult", 4))
    flash = bool(kwargs.pop("flash_attn", True))
    time_depth = int(kwargs.pop("time_transformer_depth", 1))
    freq_depth = int(kwargs.pop("freq_transformer_depth", 1))

    model = AxialModel(
        dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        ff_mult=ff_mult,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        flash_attn=flash,
        time_transformer_depth=time_depth,
        freq_transformer_depth=freq_depth,
        **kwargs,
    )
    del model.layers  # the stock pair stack is replaced by ``axial``

    if spec in ("convffn", "tailtime"):
        ffn_time = ConvFFN(dim, ff_mult, k=int(conv_k), gated=bool(conv_gated))
    else:
        ffn_time = _FFNStack(time_depth, dim, ff_mult, ff_dropout)
    ffn_every = max(1, int(ffn_every))
    model.axial = ModuleList(
        AxialBlock(
            _AxisStack(time_depth, dim, heads, dim_head, attn_dropout, flash, dim_head),
            _AxisStack(freq_depth, dim, heads, dim_head, attn_dropout, flash, dim_head),
            ffn_time if i % ffn_every == 0 else None,
            None if spec in ("singleffn", "convffn", "tailtime") else _FFNStack(freq_depth, dim, ff_mult, ff_dropout),
        )
        for i in range(depth)
    )
    if spec == "tailtime":
        # one global time-attention pass at the END: O(T^2 d) paid once instead
        # of per-block, on top of the conv-FFN's local time context.
        model.axial.append(AxialBlock(
            _AxisStack(1, dim, heads, dim_head, attn_dropout, flash, dim_head),
            nn.Identity(),
            None,
            None,
        ))
    model.spec = spec
    return model


# ---------------------------------------------------------------------------
# round-3 structures: band-axis pyramid and time-axis dual tier
#
# Both are ordinary blocks in ``AxialModel.axial`` that map (b, t, f, d) back to
# (b, t, f, d) with the model's original f and d, so ``final_norm`` and the stock
# mask estimators -- which are bound to the original 60-band layout -- keep
# working untouched.
# ---------------------------------------------------------------------------


class BandMerge(Module):
    """(b,t,n,d_in) -> (b,t,n/k,d_out), linearising each group of k neighbours."""

    def __init__(self, n_in: int, k: int, d_in: int, d_out: int):
        super().__init__()
        self.k = int(k)
        self.linear = nn.Linear(self.k * int(d_in), int(d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = x.shape
        return self.linear(x.reshape(b, t, n // self.k, self.k * x.shape[-1]))


class BandSplitBack(Module):
    """Exact inverse of :class:`BandMerge`: (b,t,n,d) -> (b,t,n*k,d_out)."""

    def __init__(self, n_in: int, k: int, d_in: int, d_out: int):
        super().__init__()
        self.k, self.d_out = int(k), int(d_out)
        self.linear = nn.Linear(int(d_in), self.k * self.d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = x.shape
        return self.linear(x).reshape(b, t, n * self.k, self.d_out)


class BandPyramidStack(Module):
    """Coarse-to-fine in the band axis: merge with depth, then restore.

    Same idea as a CNN feature pyramid, applied to (band, token) instead of
    (space, channel): early stages keep all 60 bands so frequency detail is not
    lost where it matters, later stages model fewer, wider bands for semantics.
    """

    def __init__(self, stages: ModuleList, merges: ModuleList, unmerges: ModuleList):
        super().__init__()
        self.stages, self.merges, self.unmerges = stages, merges, unmerges

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, stage in enumerate(self.stages):
            if i:
                x = self.merges[i - 1](x)
            for block in stage:
                x = block(x)
        for unmerge in self.unmerges:
            x = unmerge(x)
        return x


class TemporalTier(Module):
    """Time-axis dual tier: full-rate local branch + pooled global branch.

    The global branch runs the same blocks on a ``pool``x temporally decimated
    signal, so its cost is ``1/pool`` of an equal-length local branch.  ``gamma``
    starts at zero, so the arm begins exactly at the local-only model and the
    global branch has to earn its contribution.
    """

    def __init__(self, local: ModuleList, glob: ModuleList, pool: int = 4):
        super().__init__()
        self.local, self.glob, self.pool = local, glob, int(pool)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f, d = x.shape
        a = x
        for block in self.local:
            a = block(a)
        p = self.pool
        pad = (-t) % p
        xp = F.pad(x, (0, 0, 0, 0, 0, pad)) if pad else x
        tp = xp.shape[1]
        g = F.avg_pool1d(xp.reshape(b, tp, f * d).transpose(1, 2), p).transpose(1, 2).reshape(b, tp // p, f, d)
        for block in self.glob:
            g = block(g)
        u = F.interpolate(g.reshape(b, tp // p, f * d).transpose(1, 2), size=tp, mode="nearest")
        u = u.transpose(1, 2).reshape(b, tp, f, d)
        if pad:
            u = u[:, :t]
        return a + self.gamma * u


def _stage_blocks(dim, n_blocks, single_ffn, ff_mult, heads, dim_head) -> ModuleList:
    return ModuleList(
        AxialBlock(
            _AxisStack(1, dim, heads, dim_head, 0.0, True, dim_head),
            _AxisStack(1, dim, heads, dim_head, 0.0, True, dim_head),
            _FFNStack(1, dim, ff_mult, 0.0),
            None if single_ffn else _FFNStack(1, dim, ff_mult, 0.0),
        )
        for _ in range(n_blocks)
    )


def _make_pyramid(n_bands: int, stages, single_ffn: bool, ff_mult: int = 4) -> BandPyramidStack:
    """``stages`` is ``((dim, n_blocks), ...)``; band counts halve from ``n_bands``."""
    counts = [int(n_bands)]
    for _ in range(len(stages) - 1):
        if counts[-1] % 2:
            raise ValueError(f"cannot halve {counts[-1]} bands; use fewer stages")
        counts.append(counts[-1] // 2)
    layout = [(counts[i], stages[i][0], stages[i][1]) for i in range(len(stages))]
    stage_blocks, merges, unmerges = ModuleList(), ModuleList(), ModuleList()
    for i, (nb, dim, n_blocks) in enumerate(layout):
        if dim % 4:
            raise ValueError(f"stage dim {dim} must be divisible by 4")
        stage_blocks.append(_stage_blocks(dim, n_blocks, single_ffn, ff_mult, 4, dim // 4))
        if i:
            prev_bands, prev_dim, _ = layout[i - 1]
            merges.append(BandMerge(prev_bands, prev_bands // nb, prev_dim, dim))
    for i in range(len(layout) - 1, 0, -1):
        nb, dim, _ = layout[i]
        prev_bands, prev_dim, _ = layout[i - 1]
        unmerges.append(BandSplitBack(nb, prev_bands // nb, dim, prev_dim))
    return BandPyramidStack(stage_blocks, merges, unmerges)


def _build_compound(spec, *, dim, bands, layout, tier, single_ffn, cfg):
    heads = int(cfg.get("heads", 8))
    dim_head = int(cfg.get("dim_head", 64))
    if spec == "pyramid":
        layout = tuple(tuple(int(v) for v in stage) for stage in (layout or DEFAULT_PYRAMID))
        first_dim = layout[0][0]
    else:
        tier = tuple(int(v) for v in (tier or DEFAULT_TIER))
        first_dim = dim
    model = build_spec("baseline", dim=first_dim, bands=bands, **cfg)
    if spec == "pyramid":
        model.axial = ModuleList([_make_pyramid(len(model.band_split.dim_inputs), layout, single_ffn)])
    else:
        local, glob, pool = (tier + (4, 4, 4))[:3]
        model.axial = ModuleList([
            TemporalTier(
                _stage_blocks(first_dim, local, single_ffn, 4, heads, dim_head),
                _stage_blocks(first_dim, glob, single_ffn, 4, heads, dim_head),
                pool,
            )
        ])
    model.spec = spec
    return model
