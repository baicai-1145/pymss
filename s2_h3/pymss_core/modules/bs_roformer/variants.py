"""Architecture ablation variants for BS-RoFormer.

Every variant reuses the stock pipeline end to end -- STFT -> band split ->
axial (time/freq) blocks -> mask estimator -> complex mask -> iSTFT -- so that an
SDR delta is attributable to exactly one architectural change and not to a
different signal path, normalisation or data pipeline.

Variants (``variant=`` in the model config):

``baseline``  stock :class:`BSRoformer` (reference arm).
``bandconv``  freq-side ``FeedForward`` replaced by :class:`BandConvFFN`
              (depthwise conv along the band axis + 2x MLP instead of a second
              4x MLP): ``2*d*k + 4d^2`` per token instead of ``8d^2``.
``dilated``   time-side ``Attention`` replaced by :class:`DilatedMixer` (one 1x1
              expand, four dilated depthwise convs, one 1x1 project): drops the
              ``O(T^2)`` term and buys a fixed ``O(sum(dilation))`` receptive
              field, so long context costs nothing.
``mod``       every ``FeedForward`` wrapped in :class:`MoDFeedForward`
              (mixture-of-depths: only the top-``capacity`` fraction of tokens
              pay for the FFN).
``dualtier``  :class:`DualTierBSRoformer` -- bands split at ``split_bins`` into a
              full-width full-depth core tier and a narrow shallow tail tier,
              with cheap cross-tier conditioning and fusion.

``bandconv``/``dilated``/``mod`` are applied to an already-built model by
:func:`apply_variant`, so every module that is *not* swapped keeps its original
state-dict key and a stock checkpoint can warm-start them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList

from .bands import BandSplit, MaskEstimator
from .bs_roformer import BSRoformer
from .common import (
    DEFAULT_FREQS_PER_BANDS,
    RMSNorm,
    RotaryEmbedding,
    mask_to_complex_shape,
    roformer_freqs_per_bands_with_complex,
    roformer_stft_freq_bins,
    roformer_transformer_kwargs,
)
from .transformer import Transformer

__all__ = [
    "BandConvFFN",
    "DilatedMixer",
    "MoDFeedForward",
    "DualTierBSRoformer",
    "apply_variant",
    "build_variant",
    "mod_aux_loss",
    "reset_mod_aux",
    "VARIANT_NAMES",
]

VARIANT_NAMES = ("baseline", "bandconv", "dilated", "mod", "dualtier")


# --------------------------------------------------------------------------- #
# replacement blocks
# --------------------------------------------------------------------------- #


class BandConvFFN(Module):
    """Drop-in ``FeedForward`` replacement for the frequency (band) axis.

    Bands are spectrally smooth, so a depthwise conv along the band axis plus a
    narrower MLP buys the inter-band mixing that the second 4x MLP was paying
    ``4d^2`` for.

    Pre-normed and non-residual, matching ``FeedForward``: the parent
    ``Transformer`` adds the residual (``x = ff(x) + x``).
    """

    def __init__(self, dim: int, mult: int = 2, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.pw = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)  # (b, d, n) -- n is the band axis
        y = y + self.dw(y)
        return self.pw(y.transpose(1, 2))


class DilatedMixer(Module):
    """Drop-in ``Attention`` replacement for the time axis.

    One 1x1 expansion feeds four depthwise convs with dilations ``(1, 2, 4, 8)``;
    their sum is gated and projected back.  Cost per token is roughly
    ``3*d^2 + 4*d*k`` against attention's ``4*d^2 + 2*T*d``, while the receptive
    field is ``1 + sum(dilation) * (k-1)`` frames -- 451 frames (~4.5 s at hop
    441) for the defaults, independent of sequence length.

    Pre-normed and non-residual, matching ``Attention``.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 31,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(dilations)
        self.expand = nn.Conv1d(dim, 2 * dim, 1)
        self.depthwise = ModuleList(
            nn.Conv1d(
                dim,
                dim,
                self.kernel_size,
                padding=dilation * (self.kernel_size // 2),
                dilation=dilation,
                groups=dim,
            )
            for dilation in self.dilations
        )
        self.drop = nn.Dropout(dropout)
        self.project = nn.Linear(dim, dim)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(d * (self.kernel_size - 1) for d in self.dilations)

    # the stock Attention exposes these; keep callers that poke them happy
    def set_mps_attention_backend(self, *_args, **_kwargs) -> None:
        return None

    def set_cuda_attention_backend(self, *_args, **_kwargs) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(self.norm(x).transpose(1, 2))  # (b, 2d, n)
        gate, value = y.chunk(2, dim=1)
        mixed = value
        for conv in self.depthwise:
            mixed = mixed + conv(value)
        y = torch.sigmoid(gate) * mixed
        return self.project(self.drop(y.transpose(1, 2)))


class MoDFeedForward(Module):
    """Mixture-of-depths wrapper around a ``FeedForward``.

    A per-token scalar router selects the top ``capacity`` fraction; only those
    tokens are evaluated.  Skipped tokens emit zeros, so the parent
    ``Transformer`` residual carries them through unchanged and the shape
    contract is identical to the wrapped module.

    ``topk`` is not differentiable, so :meth:`aux_loss` supplies the standard
    MoD binary cross-entropy between the router logits and the hard decision.
    """

    def __init__(self, ffn: Module, dim: int, capacity: float = 0.5):
        super().__init__()
        if not 0.0 < capacity <= 1.0:
            raise ValueError(f"capacity must be in (0, 1], got {capacity}")
        self.ffn = ffn
        self.norm = RMSNorm(dim)
        self.router = nn.Linear(dim, 1)
        self.capacity = float(capacity)
        self._logits: list[torch.Tensor] = []
        self._keep: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        flat = x.reshape(-1, d)
        logits = self.router(self.norm(x)).reshape(-1)
        n_keep = max(1, min(flat.shape[0], int(round(self.capacity * flat.shape[0]))))
        idx = torch.topk(logits, n_keep, sorted=False).indices
        out = flat.new_zeros(flat.shape)
        out.index_copy_(0, idx, self.ffn(flat.index_select(0, idx)))
        if self.training:
            keep = torch.zeros_like(logits)
            keep[idx] = 1.0
            self._logits.append(logits)
            self._keep.append(keep)
        return out.view(b, n, d)

    def aux_loss(self) -> torch.Tensor | None:
        if not self._logits:
            return None
        loss = torch.stack(
            [F.binary_cross_entropy_with_logits(logits, keep) for logits, keep in zip(self._logits, self._keep)]
        ).mean()
        return loss * 1e-2  # standard MoD auxiliary weight

    def reset_aux(self) -> None:
        self._logits.clear()
        self._keep.clear()


# --------------------------------------------------------------------------- #
# apply variants to an already-built baseline model
# --------------------------------------------------------------------------- #


def _axial(time_tf: Module, freq_tf: Module, x: torch.Tensor) -> torch.Tensor:
    """Same factorisation as ``forward_roformer_mask_core``; ``x`` is (b, t, f, d)."""
    b, t, f, d = x.shape
    x = time_tf(x.permute(0, 2, 1, 3).reshape(b * f, t, d)).reshape(b, f, t, d).permute(0, 2, 1, 3)
    return freq_tf(x.reshape(b * t, f, d)).reshape(b, t, f, d)


def apply_variant(
    model: BSRoformer,
    variant: str,
    *,
    mod_capacity: float = 0.5,
    bandconv_mult: int = 2,
    bandconv_kernel: int = 5,
) -> BSRoformer:
    """Swap block internals in place for ``bandconv`` / ``dilated`` / ``mod``.

    Replacement happens through ``ModuleList`` item assignment, so the
    ``layers.{i}.{0,1}.layers.{j}.{0,1}`` key layout of every module that is not
    swapped is untouched.
    """
    variant = str(variant).lower()
    if variant == "baseline":
        return model
    if variant == "dualtier":
        raise ValueError("dualtier is a separate class; use build_variant()")

    swapped = 0
    for pair in model.layers:
        time_tf, freq_tf = pair[0], pair[1]
        if variant == "bandconv":
            for j, layer in enumerate(freq_tf.layers):
                dim = layer[1].net[1].in_features
                freq_tf.layers[j][1] = BandConvFFN(dim, mult=bandconv_mult, kernel_size=bandconv_kernel)
                swapped += 1
        elif variant == "dilated":
            for j, layer in enumerate(time_tf.layers):
                attn = layer[0]
                if isinstance(attn, DilatedMixer):
                    continue
                new = DilatedMixer(attn.to_qkv.in_features)
                if hasattr(attn, "mps_mlx_min_tokens"):
                    new.mps_mlx_min_tokens = attn.mps_mlx_min_tokens
                time_tf.layers[j][0] = new
                swapped += 1
        elif variant == "mod":
            for tf in (time_tf, freq_tf):
                for j, layer in enumerate(tf.layers):
                    ffn = layer[1]
                    if isinstance(ffn, MoDFeedForward):
                        continue
                    tf.layers[j][1] = MoDFeedForward(ffn, ffn.net[1].in_features, capacity=mod_capacity)
                    swapped += 1
        else:
            raise ValueError(f"unknown variant {variant!r}")

    if swapped == 0 and variant != "baseline":
        raise RuntimeError(f"variant {variant!r} swapped zero blocks; model layout unexpected")
    model.ablation_variant = variant
    return model


def mod_aux_loss(model: Module) -> torch.Tensor | None:
    """Sum of MoD router aux losses (``None`` when the model is not ``mod``)."""
    losses = [m.aux_loss() for m in model.modules() if isinstance(m, MoDFeedForward)]
    losses = [loss for loss in losses if loss is not None]
    return sum(losses) if losses else None


def reset_mod_aux(model: Module) -> None:
    for m in model.modules():
        if isinstance(m, MoDFeedForward):
            m.reset_aux()


# --------------------------------------------------------------------------- #
# dual-tier
# --------------------------------------------------------------------------- #


class DualTierBSRoformer(BSRoformer):
    """Core/tail band split with per-tier width, depth and head count.

    ``split_bins`` is a *bin* boundary, not a band index: every band whose
    cumulative bin offset is ``<= split_bins`` belongs to the core tier (so the
    actual boundary snaps to the nearest band edge, exposed as
    ``self.core_bands``).

    The tail runs ``tier2_depth`` blocks at ``tier2_dim`` and is conditioned each
    block on a pooled core summary; the core receives one low-rank fused tail
    summary before mask estimation.  Both tiers use the stock ``MaskEstimator``
    and the mask halves are concatenated in band order, so the complex-mask and
    iSTFT path is byte-for-byte the baseline one.
    """

    def __init__(
        self,
        dim: int,
        *,
        split_bins: int = 96,
        tier2_dim: int | None = None,
        tier2_depth: int | None = None,
        tier2_heads: int | None = None,
        **kwargs,
    ):
        super().__init__(dim, **kwargs)

        bands = tuple(kwargs.get("freqs_per_bands") or DEFAULT_FREQS_PER_BANDS)
        depth = int(kwargs["depth"])
        heads = int(kwargs.get("heads", 8))
        dim_head = int(kwargs.get("dim_head", 64))
        num_stems = int(kwargs.get("num_stems", 1))

        self.tier2_dim = int(tier2_dim or max(dim_head, dim // 2))
        self.tier2_depth = int(tier2_depth or max(1, depth // 2))
        self.tier2_heads = int(tier2_heads or max(1, heads // 2))
        self.split_bins = int(split_bins)

        cumulative, acc = [], 0
        for width in bands:
            acc += width
            cumulative.append(acc)
        core_bands = sum(1 for c in cumulative if c <= split_bins)
        if core_bands < 1 or core_bands >= len(bands):
            raise ValueError(f"split_bins={split_bins} leaves {core_bands}/{len(bands)} core bands")
        self.core_bands = core_bands
        self.ablation_variant = "dualtier"

        freqs = roformer_stft_freq_bins(self, self.stft_kwargs["win_length"])
        dim_inputs = roformer_freqs_per_bands_with_complex(self, bands, freqs)
        core_inputs, tail_inputs = tuple(dim_inputs[:core_bands]), tuple(dim_inputs[core_bands:])
        self._split_offset = sum(core_inputs)

        # rebuild the band modules with the tier split
        del self.band_split
        self.band_split = BandSplit(dim=dim, dim_inputs=core_inputs)
        self.band_split_tail = BandSplit(dim=self.tier2_dim, dim_inputs=tail_inputs)
        self.final_norm = RMSNorm(dim)
        self.final_norm_tail = RMSNorm(self.tier2_dim)

        tail_dim_head = max(16, self.tier2_dim // self.tier2_heads)
        tail_kwargs = roformer_transformer_kwargs(
            dim=self.tier2_dim,
            heads=self.tier2_heads,
            dim_head=tail_dim_head,
            attn_dropout=float(kwargs.get("attn_dropout", 0.0)),
            ff_dropout=float(kwargs.get("ff_dropout", 0.0)),
            flash_attn=bool(kwargs.get("flash_attn", True)),
            norm_output=False,
        )
        tail_rotary_time = RotaryEmbedding(dim=tail_dim_head)
        tail_rotary_freq = RotaryEmbedding(dim=tail_dim_head)
        self.tail_layers = ModuleList(
            ModuleList(
                [
                    Transformer(
                        depth=int(kwargs.get("time_transformer_depth", 2)),
                        rotary_embed=tail_rotary_time,
                        **tail_kwargs,
                    ),
                    Transformer(
                        depth=int(kwargs.get("freq_transformer_depth", 2)),
                        rotary_embed=tail_rotary_freq,
                        **tail_kwargs,
                    ),
                ]
            )
            for _ in range(self.tier2_depth)
        )

        # cross-tier paths (cheap, low rank)
        self.core_pool = nn.Linear(dim, self.tier2_dim)
        self.cond_proj = nn.Linear(self.tier2_dim, self.tier2_dim)
        self.tail_to_core = nn.Linear(self.tier2_dim, dim)

        # per-tier mask estimators replace the stock full-band ones
        del self.mask_estimators
        mlp_expansion_factor = int(kwargs.get("mlp_expansion_factor", 4))
        mask_depth = int(kwargs.get("mask_estimator_depth", 2))
        self.mask_estimators_core = ModuleList(
            MaskEstimator(
                dim=dim,
                dim_inputs=core_inputs,
                depth=mask_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )
            for _ in range(num_stems)
        )
        self.mask_estimators_tail = ModuleList(
            MaskEstimator(
                dim=self.tier2_dim,
                dim_inputs=tail_inputs,
                depth=mask_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )
            for _ in range(num_stems)
        )

    def _active_source_count(self) -> int:
        return len(self.mask_estimators_core)

    def _warm_group_cache(self, tensor):
        key = (tensor.device.type, tensor.device.index, tensor.dtype)
        if getattr(self, "_pymss_group_cache_warm_key", None) == key:
            return
        self.band_split.warm_group_cache(tensor.device, tensor.dtype)
        self.band_split_tail.warm_group_cache(tensor.device, tensor.dtype)
        self._pymss_group_cache_warm_key = key

    def _estimate_masks(self, x):  # pragma: no cover
        raise NotImplementedError("DualTierBSRoformer estimates masks per tier in _forward_mask_core")

    def _forward_mask_core(self, stft_repr: torch.Tensor) -> torch.Tensor:
        b, fs, model_t, complex_dim = stft_repr.shape
        features = stft_repr.permute(0, 2, 1, 3).reshape(b, model_t, fs * complex_dim)
        weights = next(self.band_split.parameters(), None)
        if weights is not None:
            features = features.to(dtype=weights.dtype)

        core = self.band_split(features[..., : self._split_offset])
        tail = self.band_split_tail(features[..., self._split_offset :])

        for i, pair in enumerate(self.layers):
            core = _axial(pair[0], pair[1], core)
            if i < len(self.tail_layers):
                tail_pair = self.tail_layers[i]
                tail = _axial(tail_pair[0], tail_pair[1], tail)
                summary = core.mean(dim=2)  # (b, t, dim)
                tail = tail + self.cond_proj(self.core_pool(summary))[:, :, None, :]

        core = core + self.tail_to_core(tail.mean(dim=2, keepdim=True))

        core, tail = self.final_norm(core), self.final_norm_tail(tail)
        masks = [
            torch.cat([est_c(core), est_t(tail)], dim=-1)
            for est_c, est_t in zip(self.mask_estimators_core, self.mask_estimators_tail)
        ]
        return mask_to_complex_shape(torch.stack(masks, dim=1), complex_dim=2)


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #


def build_variant(
    variant: str = "baseline",
    *,
    mod_capacity: float = 0.5,
    bandconv_mult: int = 2,
    bandconv_kernel: int = 5,
    **cfg,
) -> BSRoformer:
    """Construct one ablation arm; ``cfg`` is forwarded to ``BSRoformer``."""
    variant = str(variant).lower()
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANT_NAMES}")
    if variant == "dualtier":
        return DualTierBSRoformer(**cfg)
    model = BSRoformer(**cfg)
    return apply_variant(
        model,
        variant,
        mod_capacity=mod_capacity,
        bandconv_mult=bandconv_mult,
        bandconv_kernel=bandconv_kernel,
    )


# --------------------------------------------------------------------------- #
# stacking (several accelerations on one model)
# --------------------------------------------------------------------------- #


def _all_transformer_pairs(model: Module):
    """Axial pairs for a stock model and for a dual-tier model (both tiers)."""
    pairs = [(pair[0], pair[1]) for pair in model.layers]
    pairs += [(pair[0], pair[1]) for pair in getattr(model, "tail_layers", [])]
    return pairs


def _ffn_dim(ffn: Module) -> int:
    """Inner width of a FeedForward-shaped module, whatever it has been wrapped in."""
    if isinstance(ffn, MoDFeedForward):
        return _ffn_dim(ffn.ffn)
    if isinstance(ffn, BandConvFFN):
        return ffn.pw[0].in_features
    return ffn.net[1].in_features


def apply_stack(
    model: Module,
    stack,
    *,
    mod_capacity: float = 0.5,
    bandconv_mult: int = 2,
    bandconv_kernel: int = 5,
    dilations: tuple[int, ...] = (1, 2, 4, 8),
    dilated_kernel: int = 31,
) -> Module:
    """Apply several block swaps to one model, e.g. ``["mod", "bandconv"]``.

    Works on a stock ``BSRoformer`` and on ``DualTierBSRoformer`` (swapping in
    both the core and the tail tier, each at its own width).
    """
    swaps = [s for s in stack if s]
    unknown = set(swaps) - {"bandconv", "dilated", "mod"}
    if unknown:
        raise ValueError(f"unknown stack entries {sorted(unknown)}")

    counts = {"bandconv": 0, "dilated": 0, "mod": 0}
    for time_tf, freq_tf in _all_transformer_pairs(model):
        for tf in (time_tf, freq_tf):
            for j, layer in enumerate(tf.layers):
                if "bandconv" in swaps and tf is freq_tf:
                    ffn = layer[1]
                    inner = ffn.ffn if isinstance(ffn, MoDFeedForward) else ffn
                    if not isinstance(inner, BandConvFFN):
                        new = BandConvFFN(
                            _ffn_dim(inner), mult=bandconv_mult, kernel_size=bandconv_kernel
                        )
                        if isinstance(ffn, MoDFeedForward):
                            ffn.ffn = new
                        else:
                            tf.layers[j][1] = new
                        counts["bandconv"] += 1
                if "mod" in swaps:
                    ffn = tf.layers[j][1]
                    if not isinstance(ffn, MoDFeedForward):
                        tf.layers[j][1] = MoDFeedForward(ffn, _ffn_dim(ffn), capacity=mod_capacity)
                        counts["mod"] += 1
                if "dilated" in swaps and tf is time_tf:
                    attn = layer[0]
                    if not isinstance(attn, DilatedMixer):
                        new = DilatedMixer(attn.to_qkv.in_features, kernel_size=dilated_kernel, dilations=dilations)
                        tf.layers[j][0] = new
                        counts["dilated"] += 1

    if swaps and sum(counts.values()) == 0:
        raise RuntimeError(f"stack {swaps} swapped nothing")
    model.ablation_stack = tuple(swaps)
    model.ablation_counts = counts
    return model


def build_stack(
    spec: str,
    *,
    dim: int,
    mod_capacity: float = 0.5,
    bandconv_mult: int = 2,
    bandconv_kernel: int = 5,
    **cfg,
) -> Module:
    """Build from a spec string like ``"baseline"``, ``"dualtier:96+mod"``."""
    parts = [p for p in spec.replace(",", "+").split("+") if p]
    ctor_kwargs = dict(cfg)
    base = "baseline"
    for part in parts:
        if part.startswith("dualtier"):
            base = "dualtier"
            if ":" in part:
                ctor_kwargs["split_bins"] = int(part.split(":", 1)[1])
    model = DualTierBSRoformer(dim, **ctor_kwargs) if base == "dualtier" else BSRoformer(dim, **ctor_kwargs)
    return apply_stack(
        model,
        [p.split(":")[0] for p in parts if not p.startswith("dualtier") and p != "baseline"],
        mod_capacity=mod_capacity,
        bandconv_mult=bandconv_mult,
        bandconv_kernel=bandconv_kernel,
    )
