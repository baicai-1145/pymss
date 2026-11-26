from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pymss_core import to_plain


def export_core_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    model_type: str,
    config: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = _unwrap_model(model).state_dict()
    state_dict = _unswap_swiglu(state_dict, getattr(_unwrap_model(model), 'swiglu_permuted', False))
    package: dict[str, Any] = {
        "state_dict": {key: value.detach().cpu() for key, value in state_dict.items()},
        "model_type": model_type,
    }
    if config is not None:
        package["config"] = to_plain(config)
    if metadata:
        package["metadata"] = metadata

    torch.save(package, output_path)
    return output_path


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def _unswap_swiglu(state_dict: dict[str, Any], swapped: bool) -> dict[str, Any]:
    """Undo the `npu_swiglu` re-parameterisation before writing eval weights.

    `entry.prepare_swiglu` swaps the two output halves of every `MSC.proj_in` so that the fused CANN gate computes
    `u * silu(v)`.  That is a training-side convention, so exported weights are converted back here: every consumer
    (CPU fp32 eval, CUDA, the ModelScope copies) keeps loading the original layout.
    """
    if not swapped:
        return state_dict
    out = dict(state_dict)
    for key, value in state_dict.items():
        if not (key.endswith(".proj_in.weight") or key.endswith(".proj_in.bias")):
            continue
        if value is None or value.shape[0] % 2:
            continue
        half = value.shape[0] // 2
        unswapped = value.detach().clone()
        unswapped[:half] = value[half:]
        unswapped[half:] = value[:half]
        out[key] = unswapped
    return out
