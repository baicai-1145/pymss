"""Identify model architectures from YAML configuration metadata and structure."""

from collections.abc import Mapping


class ModelTypeDetectionError(RuntimeError):
    """The model configuration does not identify one supported architecture."""


_MODEL_TYPES = frozenset({
    "bs_roformer", "bs_roformer_hyperace", "bs_conformer",
    "mel_band_roformer", "mel_band_conformer", "htdemucs", "mdx23c",
    "apollo", "bandit", "bandit_v2", "scnet", "vr",
})


def detect_model_type(config):
    """Return an architecture from a loaded config, or raise ModelTypeDetectionError.

    Explicit architecture metadata takes precedence over structural hints.
    Conflicting declarations or structural hints are rejected. No weights are
    loaded; BS-RoFormer/HyperACE checkpoint refinement belongs to the runtime.
    """
    if not isinstance(config, Mapping):
        raise ModelTypeDetectionError("Cannot determine model_type: the YAML root must be a mapping. Set model_type explicitly.")

    declared = set()
    for path in (
        ("model_type",), ("model", "type"), ("model", "model_type"),
        ("model", "architecture"), ("training", "model_type"),
    ):
        value = config
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise ModelTypeDetectionError(f"Invalid architecture metadata at {'.'.join(path)}. Set model_type explicitly.")
        value = value.strip().lower()
        if value in {"", "auto"}:
            continue
        if value not in _MODEL_TYPES:
            raise ModelTypeDetectionError(f"Unsupported architecture in {'.'.join(path)}: {value!r}. Set model_type explicitly.")
        declared.add(value)

    # HTDemucs uses a scalar model name and a separate architecture config block.
    model = config.get("model", {})
    if isinstance(model, str) and model.strip().lower() == "htdemucs":
        declared.add("htdemucs")
    if len(declared) > 1:
        raise ModelTypeDetectionError(f"Conflicting model_type declarations: {', '.join(sorted(declared))}. Set model_type explicitly.")
    if declared:
        return next(iter(declared))

    candidates = set()
    if isinstance(model, Mapping):
        keys = model.keys()
        conformer_depths = "time_conformer_depth" in keys or "freq_conformer_depth" in keys
        if "freqs_per_bands" in keys:
            candidates.add("bs_conformer" if conformer_depths else "bs_roformer")
        if "num_bands" in keys:
            candidates.add("mel_band_conformer" if conformer_depths else "mel_band_roformer")
        if {"num_subbands", "num_scales", "num_blocks_per_scale", "bottleneck_factor"} <= keys:
            candidates.add("mdx23c")
        if {"band_SR", "band_stride", "band_kernel"} <= keys or {"dims", "num_dplayer"} <= keys:
            candidates.add("scnet")
        if {"sr", "win", "feature_dim", "layer"} <= keys:
            candidates.add("apollo")
        if {"in_channel", "stems", "band_specs"} <= keys:
            candidates.add("bandit")
    kwargs = config.get("kwargs", {})
    if isinstance(kwargs, Mapping) and {"in_channels", "stems"} <= kwargs.keys():
        candidates.add("bandit_v2")

    if len(candidates) == 1:
        return next(iter(candidates))
    if candidates:
        raise ModelTypeDetectionError(f"Ambiguous model architecture: {', '.join(sorted(candidates))}. Set model_type explicitly.")
    raise ModelTypeDetectionError("Cannot determine model_type from the configuration. Set model_type explicitly.")
