"""Torch-version compatibility helpers."""
from __future__ import annotations

import torch

# torch 1.10's inference_mode tensors break custom backends
# ("Cannot set version_counter for inference tensor" on GCU).
_INFER_CTX = torch.inference_mode if float(torch.__version__.split("+")[0].rsplit(".", 1)[0]) >= 1.12 else torch.no_grad


def inference_ctx():
    """Return inference_mode on modern torch, no_grad on old builds."""
    return _INFER_CTX()
