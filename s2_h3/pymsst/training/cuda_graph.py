from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass
class CudaGraphRuntimeStatus:
    requested: str
    active: str
    reason: str | None = None


def normalize_cuda_graph_mode(value: str | None) -> str:
    mode = str(value or "off").strip().lower().replace("-", "_")
    if mode in {"", "none", "false", "0", "off", "default"}:
        return "off"
    if mode in {"mask", "mask_core", "roformer_mask_core"}:
        return "mask_core"
    if mode in {"stft_mask", "stft_mask_core", "roformer_stft_mask_core"}:
        return "stft_mask_core"
    raise ValueError("cuda_graph_mode must be one of: off, mask_core, stft_mask_core")


class RoformerTrainingCudaGraph:
    def __init__(self, model: torch.nn.Module, mode: str = "mask_core") -> None:
        self.mode = normalize_cuda_graph_mode(mode)
        if self.mode == "off":
            raise ValueError("RoformerTrainingCudaGraph requires an active graph mode")
        self.model = model
        self.status = CudaGraphRuntimeStatus(requested=self.mode, active="off")
        self._graph_call: Callable[[torch.Tensor], torch.Tensor] | None = None
        self._context = None
        self._input_shape: tuple[int, ...] | None = None

    @property
    def active(self) -> bool:
        return self.status.active == self.mode and self._graph_call is not None

    def capture(self, sample_mixture: torch.Tensor, *, use_amp: bool) -> CudaGraphRuntimeStatus:
        if sample_mixture.device.type != "cuda":
            self.status = CudaGraphRuntimeStatus(self.mode, "off", "CUDA graph requires CUDA tensors")
            return self.status

        try:
            import pymss_core.modules.bs_roformer.common as common
        except Exception as exc:
            self.status = CudaGraphRuntimeStatus(self.mode, "off", f"failed to import pymss-core common: {exc}")
            return self.status

        try:
            if self.mode == "stft_mask_core":
                with _cuda_autocast(use_amp):
                    _, context = common.stft_roformer(self.model, sample_mixture)
                    graph_module = _STFTMaskCoreModule(self.model)
                    graph_module(sample_mixture)
                    self._graph_call = torch.cuda.make_graphed_callables(graph_module, (sample_mixture,))
            else:
                with _cuda_autocast(use_amp):
                    sample_stft, context = common.stft_roformer(self.model, sample_mixture)

                graph_module = _MaskCoreModule(self.model, context)

                with _cuda_autocast(use_amp):
                    graph_module(sample_stft)
                    self._graph_call = torch.cuda.make_graphed_callables(graph_module, (sample_stft,))
            self._context = context
            self._input_shape = tuple(sample_mixture.shape)
            self.status = CudaGraphRuntimeStatus(self.mode, self.mode)
        except Exception as exc:
            self._graph_call = None
            self._context = None
            self._input_shape = None
            reason = f"failed to capture RoFormer {self.mode} CUDA graph: {exc}"
            warnings.warn(reason, RuntimeWarning, stacklevel=2)
            self.status = CudaGraphRuntimeStatus(self.mode, "off", reason)
        finally:
            _clear_grads(self.model)
        return self.status

    def forward(self, mixture: torch.Tensor, *, use_amp: bool) -> torch.Tensor:
        if not self.active:
            return self.model(mixture)
        if self._input_shape is not None and tuple(mixture.shape) != self._input_shape:
            return self.model(mixture)

        if self.mode == "stft_mask_core":
            with _cuda_autocast(use_amp):
                masked = self._graph_call(mixture)
                length = self._context.audio_length if getattr(self.model, "match_input_audio_length", True) else None
                return _istft_roformer_cached_zero(self.model, masked, self._context, length)

        import pymss_core.modules.bs_roformer.common as common

        with _cuda_autocast(use_amp):
            stft_repr, context = common.stft_roformer(self.model, mixture)
            masked = self._graph_call(stft_repr)
            length = context.audio_length if getattr(self.model, "match_input_audio_length", True) else None
            return _istft_roformer_cached_zero(self.model, masked, context, length)


def _cuda_autocast(enabled: bool):
    return torch.autocast("cuda", dtype=torch.float16, enabled=enabled, cache_enabled=False)


class _MaskCoreModule(nn.Module):
    def __init__(self, model: nn.Module, context) -> None:
        super().__init__()
        self.model = model
        self.context = context

    def forward(self, stft_repr: torch.Tensor) -> torch.Tensor:
        return self.model._mask_stft_repr(stft_repr, self.context)


class _STFTMaskCoreModule(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        import pymss_core.modules.bs_roformer.common as common

        stft_repr, context = common.stft_roformer(self.model, mixture)
        return self.model._mask_stft_repr(stft_repr, context)


RoformerMaskCoreCudaGraph = RoformerTrainingCudaGraph


def _clear_grads(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.grad = None


def _istft_roformer_cached_zero(module, stft_repr: torch.Tensor, context, length: int | None) -> torch.Tensor:
    b, n, _, t = stft_repr.shape
    stft_repr = (
        stft_repr.reshape(b, n, context.freq_bins, context.channels, t)
        .permute(0, 1, 3, 2, 4)
        .reshape(b * n * context.channels, context.freq_bins, t)
    )
    if getattr(module, "zero_dc", False):
        stft_repr = stft_repr.index_fill(1, _zero_dc_index(module, stft_repr.device), 0.0)

    recon_audio = torch.istft(
        stft_repr,
        **module.stft_kwargs,
        window=context.stft_window,
        return_complex=False,
        length=length,
    )
    recon_audio = recon_audio.reshape(context.batch, n, context.channels, recon_audio.shape[-1])
    return recon_audio[:, 0] if n == 1 else recon_audio


def _zero_dc_index(module, device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    cache = getattr(module, "_pymsst_zero_dc_index_cache", None)
    if cache is None:
        cache = {}
        module._pymsst_zero_dc_index_cache = cache
    zero = cache.get(key)
    if zero is None or zero.device != device:
        zero = torch.zeros(1, dtype=torch.long, device=device)
        cache[key] = zero
    return zero
