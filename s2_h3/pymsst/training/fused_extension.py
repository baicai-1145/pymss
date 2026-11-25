from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch


class FusedKernelUnavailable(RuntimeError):
    """Raised when the optional CUDA extension cannot be used."""


def load_fused_band_split_extension() -> ModuleType:
    if not torch.cuda.is_available():
        raise FusedKernelUnavailable("CUDA is not available")
    if torch.version.cuda is None:
        raise FusedKernelUnavailable("PyTorch was not built with CUDA")
    return _load_fused_band_split_extension_cached(_extension_verbose())


def _extension_verbose() -> bool:
    return os.environ.get("PYMSST_CUDA_EXT_VERBOSE", "").lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=2)
def _load_fused_band_split_extension_cached(verbose: bool) -> ModuleType:
    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - depends on torch packaging
        raise FusedKernelUnavailable(f"torch C++ extension loader is unavailable: {exc}") from exc

    source_dir = Path(__file__).resolve().parent / "cuda"
    sources = [
        source_dir / "fused_band_split.cpp",
        source_dir / "fused_band_split.cu",
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FusedKernelUnavailable(f"CUDA extension sources are missing: {missing}")

    build_directory = os.environ.get("PYMSST_CUDA_EXT_BUILD_DIR") or None
    try:
        return load(
            name="pymsst_fused_band_split",
            sources=[str(path) for path in sources],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_ldflags=["-lcublas"],
            build_directory=build_directory,
            verbose=verbose,
            with_cuda=True,
            keep_intermediates=False,
        )
    except Exception as exc:
        raise FusedKernelUnavailable(f"failed to build/load pymsst CUDA extension: {exc}") from exc
