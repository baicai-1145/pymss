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
