from __future__ import annotations

import os
import types
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .fused_extension import FusedKernelUnavailable, load_fused_band_split_extension


CORE_GROUPED_TRAIN_ENV = "PYMSS_CORE_EXPERIMENTAL_TRAIN_GROUPED_BANDS"
EXPERIMENTAL_GROUPED_LINEAR_ENV = "PYMSST_EXPERIMENTAL_FUSED_GROUPED_LINEAR"
EXPERIMENTAL_BAND_SPLIT_BWD_V2_ENV = "PYMSST_EXPERIMENTAL_BANDSPLIT_BWD_V2"
EXPERIMENTAL_BAND_SPLIT_BWD_V3_ENV = "PYMSST_EXPERIMENTAL_BANDSPLIT_BWD_V3"
EXPERIMENTAL_BAND_SPLIT_RAGGED_FORWARD_ENV = "PYMSST_EXPERIMENTAL_BANDSPLIT_RAGGED_FORWARD"
EXPERIMENTAL_BAND_SPLIT_RAGGED_BACKWARD_ENV = "PYMSST_EXPERIMENTAL_BANDSPLIT_RAGGED_BACKWARD"
EXPERIMENTAL_ROTARY_ENV = "PYMSST_EXPERIMENTAL_FUSED_ROTARY"
EXPERIMENTAL_RMS_NORM_ENV = "PYMSST_EXPERIMENTAL_FUSED_RMSNORM"
EXPERIMENTAL_RMS_NORM_BWD_V2_ENV = "PYMSST_EXPERIMENTAL_RMSNORM_BWD_V2"
EXPERIMENTAL_ATTENTION_GATES_ENV = "PYMSST_EXPERIMENTAL_FUSED_ATTENTION_GATES"
EXPERIMENTAL_LAYOUT_REDUCE_COPY_ENV = "PYMSST_EXPERIMENTAL_LAYOUT_REDUCE_COPY"
EXPERIMENTAL_MASK_ESTIMATOR_FINAL_ENV = "PYMSST_EXPERIMENTAL_MASK_ESTIMATOR_FINAL_FUSED"
EXPERIMENTAL_MASK_ESTIMATOR_FINAL_DIRECT_ENV = "PYMSST_EXPERIMENTAL_MASK_ESTIMATOR_FINAL_DIRECT"
EXPERIMENTAL_MASK_FINAL_AUTOGRAD_ENV = "PYMSST_EXPERIMENTAL_MASK_FINAL_AUTOGRAD"
EXPERIMENTAL_TRAIN_PARAM_PACK_AUTOGRAD_ENV = "PYMSST_EXPERIMENTAL_TRAIN_PARAM_PACK_AUTOGRAD"


@dataclass(frozen=True)
class FusedTrainingStatus:
    requested: str
    active: str
    reason: str | None = None
    patched_band_splits: int = 0
    patched_band_split_bwd_v2: bool = False
    patched_band_split_bwd_v3: bool = False
    patched_band_split_ragged_forward: bool = False
    patched_band_split_ragged_backward: bool = False
    patched_rotary: bool = False
    patched_rms_norm: bool = False
    rms_norm_mode: str | None = None
    patched_rms_norm_bwd_v2: bool = False
    patched_single_stem_masks: bool = False
    patched_attention_gates: bool = False
    attention_gate_mode: str | None = None
    patched_layout_reduce_copy: bool = False
    patched_train_param_pack_autograd: bool = False
    patched_mask_final_autograd: bool = False
    patched_mask_estimator_final: bool = False
    mask_estimator_final_mode: str | None = None


def apply_roformer_fused_training_backend(
    model: torch.nn.Module,
    backend: str | None,
    *,
    extension: Any | None = None,
    rms_norm_mode: str | None = None,
) -> FusedTrainingStatus:
    normalized = normalize_fused_kernel_backend(backend)
    if normalized == "off":
        return FusedTrainingStatus(requested="off", active="off")

    if normalized == "g4":
        from importlib.util import find_spec
        if not torch.cuda.is_available() or find_spec("triton") is None:
            reason = "g4 requires CUDA and Triton"
            warnings.warn(reason, RuntimeWarning, stacklevel=2)
            return FusedTrainingStatus(requested="g4", active="off", reason=reason)
        from .g4 import install_g4
        from .g4_rope import install as install_packed_rope
        from .g4_time import install as install_time_dw
        rotary = install_packed_rope(model)
        norms = install_g4(model)
        temporal = install_time_dw(model)
        return FusedTrainingStatus(
            requested="g4", active="g4" if norms or rotary or temporal else "off",
            patched_rotary=bool(rotary), patched_rms_norm=bool(norms), rms_norm_mode="inductor" if norms else None,
        )

    if normalized != "cuda_ext":
        raise ValueError("fused_kernel_backend must be one of: off, cuda_ext, g4")

    try:
        ext = extension if extension is not None else load_fused_band_split_extension()
    except FusedKernelUnavailable as exc:
        warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
        return FusedTrainingStatus(requested=normalized, active="off", reason=str(exc))

    band_split_bwd_mode = _experimental_band_split_bwd_mode(ext)
    patched = _patch_band_split_modules(model, ext, bwd_mode=band_split_bwd_mode)
    patched_single_stem_masks = _patch_single_stem_mask_estimation()
    patched_rotary = _patch_core_rotary(ext) if _experimental_rotary_enabled() else False
    normalized_rms_norm_mode = normalize_fused_rms_norm_mode(rms_norm_mode)
    use_rms_norm_bwd_v2 = _experimental_rms_norm_bwd_v2_enabled(ext, normalized_rms_norm_mode)
    patched_rms_norm = (
        _patch_core_rms_norm(ext, normalized_rms_norm_mode, use_bwd_v2=use_rms_norm_bwd_v2)
        if normalized_rms_norm_mode
        else False
    )
    attention_gate_mode = _experimental_attention_gates_mode()
    patched_attention_gates = _patch_core_attention_gates(ext, attention_gate_mode) if attention_gate_mode else False
    patched_grouped_linear = _patch_core_grouped_linear(ext) if _experimental_grouped_linear_enabled() else False
    patched_layout_reduce_copy = _patch_core_layout_reduce_copy() if _experimental_layout_reduce_copy_enabled() else False
    patched_train_param_pack_autograd = patched_layout_reduce_copy and _experimental_train_param_pack_autograd_enabled()
    patched_mask_final_autograd = patched_layout_reduce_copy and _experimental_mask_final_autograd_enabled()
    mask_estimator_final_mode = _experimental_mask_estimator_final_mode(ext)
    patched_mask_estimator_final = _patch_core_mask_estimator_final(ext, mask_estimator_final_mode) if mask_estimator_final_mode else False
    if (
        not patched
        and not patched_single_stem_masks
        and not patched_rotary
        and not patched_rms_norm
        and not patched_attention_gates
        and not patched_grouped_linear
        and not patched_layout_reduce_copy
        and not patched_train_param_pack_autograd
        and not patched_mask_final_autograd
        and not patched_mask_estimator_final
    ):
        reason = "no compatible pymss-core BandSplit modules were found"
        warnings.warn(reason, RuntimeWarning, stacklevel=2)
        return FusedTrainingStatus(requested=normalized, active="off", reason=reason)

    os.environ.setdefault(CORE_GROUPED_TRAIN_ENV, "1")
    return FusedTrainingStatus(
        requested=normalized,
        active=normalized,
        patched_band_splits=patched,
        patched_band_split_bwd_v2=patched > 0 and band_split_bwd_mode == "v2",
        patched_band_split_bwd_v3=patched > 0 and band_split_bwd_mode == "v3",
        patched_band_split_ragged_forward=patched > 0 and _experimental_band_split_ragged_forward_enabled(ext),
        patched_band_split_ragged_backward=(
            patched > 0
            and _experimental_band_split_ragged_forward_enabled(ext)
            and _experimental_band_split_ragged_backward_enabled(ext)
        ),
        patched_rotary=patched_rotary,
        patched_rms_norm=patched_rms_norm,
        rms_norm_mode=normalized_rms_norm_mode if patched_rms_norm else None,
        patched_rms_norm_bwd_v2=patched_rms_norm and use_rms_norm_bwd_v2,
        patched_single_stem_masks=patched_single_stem_masks,
        patched_attention_gates=patched_attention_gates,
        attention_gate_mode=attention_gate_mode if patched_attention_gates else None,
        patched_layout_reduce_copy=patched_layout_reduce_copy,
        patched_train_param_pack_autograd=patched_train_param_pack_autograd,
        patched_mask_final_autograd=patched_mask_final_autograd,
        patched_mask_estimator_final=patched_mask_estimator_final,
        mask_estimator_final_mode=mask_estimator_final_mode if patched_mask_estimator_final else None,
    )


def normalize_fused_kernel_backend(value: str | None) -> str:
    backend = str(value or "off").strip().lower().replace("-", "_")
    if backend == "g4":
        return backend
    if backend in {"", "none", "false", "0", "off", "default"}:
        return "off"
    if backend in {"cuda", "cuda_ext", "cuda_extension", "cpp_cuda", "cxx_cuda"}:
        return "cuda_ext"
    raise ValueError("fused_kernel_backend must be one of: off, cuda_ext, g4")


def normalize_fused_rms_norm_mode(value: str | None) -> str | None:
    if value is None:
        mode = _experimental_rms_norm_mode()
        if mode is not None:
            return mode
        return "fast" if _experimental_rms_norm_bwd_v2_requested() else None
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"", "none", "false", "0", "off", "default"}:
        return None
    if mode in {"mixed", "fp32", "safe"}:
        return "mixed"
    if mode in {"1", "true", "yes", "on", "fast", "half", "fp16"}:
        return "fast"
    raise ValueError("fused_rms_norm_mode must be one of: off, fast, mixed")


def _experimental_grouped_linear_enabled() -> bool:
    return os.environ.get(EXPERIMENTAL_GROUPED_LINEAR_ENV, "").lower() in {"1", "true", "yes", "on"}


def _experimental_band_split_bwd_mode(extension: Any) -> str | None:
    if (
        os.environ.get(EXPERIMENTAL_BAND_SPLIT_BWD_V3_ENV, "").lower() in {"1", "true", "yes", "on"}
        and hasattr(extension, "band_split_backward_v3")
    ):
        return "v3"
    if (
        os.environ.get(EXPERIMENTAL_BAND_SPLIT_BWD_V2_ENV, "").lower() in {"1", "true", "yes", "on"}
        and hasattr(extension, "band_split_backward_v2")
    ):
        return "v2"
    return None


def _experimental_band_split_ragged_forward_enabled(extension: Any) -> bool:
    return (
        os.environ.get(EXPERIMENTAL_BAND_SPLIT_RAGGED_FORWARD_ENV, "").lower() in {"1", "true", "yes", "on"}
        and hasattr(extension, "band_split_ragged_forward")
    )


def _experimental_band_split_ragged_backward_enabled(extension: Any) -> bool:
    return (
        os.environ.get(EXPERIMENTAL_BAND_SPLIT_RAGGED_BACKWARD_ENV, "").lower() in {"1", "true", "yes", "on"}
        and hasattr(extension, "band_split_ragged_backward")
    )


def _experimental_rotary_enabled() -> bool:
    return os.environ.get(EXPERIMENTAL_ROTARY_ENV, "").lower() in {"1", "true", "yes", "on"}


def _experimental_rms_norm_mode() -> str | None:
    value = os.environ.get(EXPERIMENTAL_RMS_NORM_ENV, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return None
    if value in {"mixed", "fp32", "safe"}:
        return "mixed"
    if value in {"1", "true", "yes", "on", "fast", "half"}:
        return "fast"
    return None


def _experimental_rms_norm_bwd_v2_enabled(extension: Any, mode: str | None) -> bool:
    if mode != "fast":
        return False
    return (
        _experimental_rms_norm_bwd_v2_requested()
        and hasattr(extension, "rms_norm_forward_fp32_gamma")
        and hasattr(extension, "rms_norm_backward_fp32_gamma")
    )


def _experimental_rms_norm_bwd_v2_requested() -> bool:
    return os.environ.get(EXPERIMENTAL_RMS_NORM_BWD_V2_ENV, "").lower() in {"1", "true", "yes", "on"}


def _experimental_attention_gates_mode() -> str | None:
    value = os.environ.get(EXPERIMENTAL_ATTENTION_GATES_ENV, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return None
    if value in {"1", "true", "yes", "on", "full", "gate"}:
        return "full"
    return None


def _experimental_layout_reduce_copy_enabled() -> bool:
    return os.environ.get(EXPERIMENTAL_LAYOUT_REDUCE_COPY_ENV, "").lower() in {"1", "true", "yes", "on"}


def _experimental_mask_estimator_final_mode(extension: Any) -> str | None:
    direct = os.environ.get(EXPERIMENTAL_MASK_ESTIMATOR_FINAL_DIRECT_ENV, "").lower() in {"1", "true", "yes", "on"}
    if direct and hasattr(extension, "mask_estimator_final_direct_forward") and hasattr(extension, "mask_estimator_final_direct_backward"):
        return "direct"
    grouped = os.environ.get(EXPERIMENTAL_MASK_ESTIMATOR_FINAL_ENV, "").lower() in {"1", "true", "yes", "on"}
    if grouped and hasattr(extension, "mask_estimator_final_forward") and hasattr(extension, "mask_estimator_final_backward"):
        return "grouped"
    return None


def _experimental_mask_final_autograd_enabled() -> bool:
    return os.environ.get(EXPERIMENTAL_MASK_FINAL_AUTOGRAD_ENV, "").lower() in {"1", "true", "yes", "on"}


def _experimental_train_param_pack_autograd_enabled() -> bool:
    return os.environ.get(EXPERIMENTAL_TRAIN_PARAM_PACK_AUTOGRAD_ENV, "").lower() in {"1", "true", "yes", "on"}


def _patch_band_split_modules(model: torch.nn.Module, extension: Any, *, bwd_mode: str | None = None) -> int:
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "BandSplit":
            continue
        if not hasattr(module, "_get_group_params") or not hasattr(module, "_dim_offsets"):
            continue
        if getattr(module, "_pymsst_fused_band_split", False):
            continue
        module._pymsst_original_forward_grouped = module._forward_grouped
        module._pymsst_fused_extension = extension
        module._pymsst_fused_band_split_bwd_mode = bwd_mode
        module._forward_grouped = types.MethodType(_forward_grouped_band_split_cuda_ext, module)
        module._pymsst_fused_band_split = True
        patched += 1
    return patched


def _patch_core_grouped_linear(extension: Any) -> bool:
    if not hasattr(extension, "grouped_linear_forward"):
        return False
    try:
        import pymss_core.modules.bs_roformer.bands as bands
    except Exception:
        return False
    if getattr(bands.grouped_linear, "_pymsst_fused_grouped_linear", False):
        return True

    original_grouped_linear = bands.grouped_linear

    def grouped_linear_cuda_ext(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        if x.device.type != "cuda" or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return original_grouped_linear(x, weight, bias)
        if weight.device != x.device or weight.dtype != x.dtype or (bias is not None and (bias.device != x.device or bias.dtype != x.dtype)):
            return original_grouped_linear(x, weight, bias)
        return _FusedGroupedLinear.apply(x, weight, bias, extension)

    grouped_linear_cuda_ext._pymsst_fused_grouped_linear = True
    grouped_linear_cuda_ext._pymsst_original_grouped_linear = original_grouped_linear
    bands.grouped_linear = grouped_linear_cuda_ext
    return True


def _patch_single_stem_mask_estimation() -> bool:
    try:
        import pymss_core.modules.bs_roformer.common as common
    except Exception:
        return False
    if getattr(common.RoformerRuntimeMixin._estimate_masks, "_pymsst_single_stem_shortcut", False):
        return True

    original_estimate_masks = common.RoformerRuntimeMixin._estimate_masks

    def estimate_masks_single_stem_shortcut(self, x: torch.Tensor):
        estimators = self._active_mask_estimators()
        if len(estimators) == 1:
            return estimators[0](x).unsqueeze(1)
        return original_estimate_masks(self, x)

    estimate_masks_single_stem_shortcut._pymsst_single_stem_shortcut = True
    estimate_masks_single_stem_shortcut._pymsst_original_estimate_masks = original_estimate_masks
    common.RoformerRuntimeMixin._estimate_masks = estimate_masks_single_stem_shortcut
    return True


def _patch_core_layout_reduce_copy() -> bool:
    try:
        import pymss_core.modules.bs_roformer.bands as bands
    except Exception:
        return False
    patched = False
    already_patched = False

    if not getattr(bands.MaskEstimator._forward_layer_grouped_mlp, "_pymsst_layout_reduce_copy", False):
        original_forward_layer_grouped_mlp = bands.MaskEstimator._forward_layer_grouped_mlp

        def forward_layer_grouped_mlp_layout_reduce_copy(self, x: torch.Tensor):
            plan = self._layer_grouping_plan()
            if plan is None:
                return None

            group_x = x
            for layer_index, (kind, groups) in enumerate(plan):
                if kind == "tanh":
                    group_x = bands.inference_tanh(group_x)
                    continue

                out_dims = {signature[1] for signature, _ in groups}
                if len(out_dims) != 1:
                    if layer_index != len(plan) - 1:
                        return None
                    outs = [None] * len(self.to_freqs)
                    for signature, indices in groups:
                        weight, bias = self._get_layer_group_params(layer_index, signature, indices, x.device, x.dtype)
                        selected = _select_band_group(group_x, indices, x.device, self._indices_tensor)
                        if _experimental_mask_final_autograd_enabled():
                            out = _MaskFinalGroupedLinearGLU.apply(selected, weight, bias)
                        else:
                            out = F.glu(bands.grouped_linear(selected, weight, bias), dim=-1)
                        for band_position, band_out in zip(indices, out.unbind(dim=-2)):
                            outs[band_position] = band_out
                    return torch.cat(outs, dim=-1)

                next_x = None
                for signature, indices in groups:
                    weight, bias = self._get_layer_group_params(layer_index, signature, indices, x.device, x.dtype)
                    selected = _select_band_group(group_x, indices, x.device, self._indices_tensor)
                    out = bands.grouped_linear(selected, weight, bias)
                    if next_x is None:
                        next_x = out.new_empty(*group_x.shape[:-1], out.shape[-1])
                    _copy_band_group(next_x, indices, out, x.device, self._indices_tensor)
                group_x = next_x

            return F.glu(group_x, dim=-1).flatten(start_dim=-2)

        forward_layer_grouped_mlp_layout_reduce_copy._pymsst_layout_reduce_copy = True
        forward_layer_grouped_mlp_layout_reduce_copy._pymsst_original_forward_layer_grouped_mlp = (
            original_forward_layer_grouped_mlp
        )
        bands.MaskEstimator._forward_layer_grouped_mlp = forward_layer_grouped_mlp_layout_reduce_copy
        patched = True
    else:
        already_patched = True

    if not getattr(bands.BandSplit._get_group_params, "_pymsst_layout_reduce_copy", False):
        original_band_split_get_group_params = bands.BandSplit._get_group_params

        def band_split_get_group_params_layout_reduce_copy(self, start, end, device, dtype):
            key = (start, end, device.type, device.index, dtype)
            use_cache = not self.training
            if use_cache:
                cached = self._group_cache.get(key)
                if cached is not None:
                    return cached

            gammas, weights, biases = _band_split_group_param_refs(self, start, end)
            gamma = _stack_converted_tensors(gammas, device, dtype)
            weight = _stack_converted_tensors(weights, device, dtype)
            bias = _stack_converted_tensors(biases, device, dtype) if biases is not None else None
            cached = (gamma, weight, bias)
            if use_cache:
                self._group_cache[key] = cached
            return cached

        band_split_get_group_params_layout_reduce_copy._pymsst_layout_reduce_copy = True
        band_split_get_group_params_layout_reduce_copy._pymsst_original_get_group_params = original_band_split_get_group_params
        bands.BandSplit._get_group_params = band_split_get_group_params_layout_reduce_copy
        patched = True
    else:
        already_patched = True

    if not getattr(bands.MaskEstimator._get_group_params, "_pymsst_layout_reduce_copy", False):
        original_mask_get_group_params = bands.MaskEstimator._get_group_params

        def mask_get_group_params_layout_reduce_copy(self, start, end, device, dtype):
            key = (start, end, device.type, device.index, dtype)
            use_cache = not self.training
            if use_cache:
                cached = self._group_cache.get(key)
                if cached is not None:
                    return cached

            grouped_layers = []
            band_layers = self._band_groupable_layers()
            first_layers = band_layers[start]
            for layer_index, (kind, _) in enumerate(first_layers):
                if kind == "tanh":
                    grouped_layers.append(("tanh", None, None))
                    continue

                weights, biases = _mask_layer_param_refs(self, layer_index, tuple(range(start, end)))
                weight = _stack_converted_tensors(weights, device, dtype)
                bias = _stack_converted_tensors(biases, device, dtype) if biases is not None else None
                grouped_layers.append(("linear", weight, bias))

            cached = tuple(grouped_layers)
            if use_cache:
                self._group_cache[key] = cached
            return cached

        mask_get_group_params_layout_reduce_copy._pymsst_layout_reduce_copy = True
        mask_get_group_params_layout_reduce_copy._pymsst_original_get_group_params = original_mask_get_group_params
        bands.MaskEstimator._get_group_params = mask_get_group_params_layout_reduce_copy
        patched = True
    else:
        already_patched = True

    if not getattr(bands.MaskEstimator._get_layer_group_params, "_pymsst_layout_reduce_copy", False):
        original_get_layer_group_params = bands.MaskEstimator._get_layer_group_params

        def get_layer_group_params_layout_reduce_copy(self, layer_index, signature, indices, device, dtype):
            key = (layer_index, signature, indices, device.type, device.index, dtype)
            use_cache = not self.training
            if use_cache:
                cached = self._layer_group_cache.get(key)
                if cached is not None:
                    return cached

            weights, biases = _mask_layer_param_refs(self, layer_index, indices)
            weight = _stack_converted_tensors(weights, device, dtype)
            bias = _stack_converted_tensors(biases, device, dtype) if biases is not None else None

            cached = (weight, bias)
            if use_cache:
                self._layer_group_cache[key] = cached
            return cached

        get_layer_group_params_layout_reduce_copy._pymsst_layout_reduce_copy = True
        get_layer_group_params_layout_reduce_copy._pymsst_original_get_layer_group_params = original_get_layer_group_params
        bands.MaskEstimator._get_layer_group_params = get_layer_group_params_layout_reduce_copy
        patched = True
    else:
        already_patched = True

    if not getattr(bands.MaskEstimator._get_packed_layer_group_params, "_pymsst_layout_reduce_copy", False):
        original_get_packed_layer_group_params = bands.MaskEstimator._get_packed_layer_group_params

        def get_packed_layer_group_params_layout_reduce_copy(self, estimators, layer_index, signature, indices, device, dtype):
            estimator_ids = tuple(id(estimator) for estimator in estimators)
            key = (estimator_ids, layer_index, signature, indices, device.type, device.index, dtype)
            use_cache = not any(estimator.training for estimator in estimators)
            if use_cache:
                cached = self._packed_layer_group_cache.get(key)
                if cached is not None:
                    return cached

            weights, biases = _packed_mask_layer_param_refs(self, estimators, layer_index, indices)
            weight = _stack_converted_tensors(weights, device, dtype)
            bias = _stack_converted_tensors(biases, device, dtype) if biases is not None else None

            cached = (weight, bias)
            if use_cache:
                self._packed_layer_group_cache[key] = cached
            return cached

        get_packed_layer_group_params_layout_reduce_copy._pymsst_layout_reduce_copy = True
        get_packed_layer_group_params_layout_reduce_copy._pymsst_original_get_packed_layer_group_params = (
            original_get_packed_layer_group_params
        )
        bands.MaskEstimator._get_packed_layer_group_params = get_packed_layer_group_params_layout_reduce_copy
        patched = True
    else:
        already_patched = True

    return patched or already_patched


def _patch_core_mask_estimator_final(extension: Any, mode: str) -> bool:
    if mode == "direct":
        required = ("mask_estimator_final_direct_forward", "mask_estimator_final_direct_backward")
    else:
        required = ("mask_estimator_final_forward", "mask_estimator_final_backward")
    if any(not hasattr(extension, name) for name in required):
        return False
    if mode != "direct" and (not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9):
        warnings.warn(
            "PYMSST_EXPERIMENTAL_MASK_ESTIMATOR_FINAL_FUSED is disabled on this GPU: "
            "cuBLAS grouped/batched GEMM rejected the current FP16 ragged MBR shapes on sm86 during validation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    try:
        import pymss_core.modules.bs_roformer.bands as bands
    except Exception:
        return False
    if getattr(bands.MaskEstimator._forward_layer_grouped_mlp, "_pymsst_mask_estimator_final_fused", False):
        return True

    original_forward_layer_grouped_mlp = bands.MaskEstimator._forward_layer_grouped_mlp

    def forward_layer_grouped_mlp_mask_final_fused(self, x: torch.Tensor):
        if x.device.type != "cuda" or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return original_forward_layer_grouped_mlp(self, x)

        plan = self._layer_grouping_plan()
        if plan is None or not plan or plan[-1][0] != "linear":
            return original_forward_layer_grouped_mlp(self, x)

        group_x = x
        for layer_index, (kind, groups) in enumerate(plan[:-1]):
            if kind == "tanh":
                group_x = bands.inference_tanh(group_x)
                continue

            out_dims = {signature[1] for signature, _ in groups}
            if len(out_dims) != 1:
                return original_forward_layer_grouped_mlp(self, x)

            next_x = None
            for signature, indices in groups:
                weight, bias = self._get_layer_group_params(layer_index, signature, indices, x.device, x.dtype)
                selected = _select_band_group(group_x, indices, x.device, self._indices_tensor)
                out = bands.grouped_linear(selected, weight, bias)
                if next_x is None:
                    next_x = out.new_empty(*group_x.shape[:-1], out.shape[-1])
                _copy_band_group(next_x, indices, out, x.device, self._indices_tensor)
            group_x = next_x

        band_layers = self._band_groupable_layers()
        final_weights = []
        final_biases = []
        final_layer_index = len(plan) - 1
        for band_index, layers in enumerate(band_layers):
            kind, layer = layers[final_layer_index]
            if kind != "linear":
                return original_forward_layer_grouped_mlp(self, x)
            if layer.in_features != group_x.shape[-1] or layer.out_features != self.dim_inputs[band_index] * 2:
                return original_forward_layer_grouped_mlp(self, x)
            final_weights.append(layer.weight.to(device=group_x.device, dtype=group_x.dtype))
            if layer.bias is None:
                return original_forward_layer_grouped_mlp(self, x)
            final_biases.append(layer.bias.to(device=group_x.device, dtype=group_x.dtype))

        dim_offsets = _mask_estimator_final_dim_offsets(self, x.device)
        leading_shape = group_x.shape[:-2]
        group_x = group_x.reshape(-1, len(self.to_freqs), group_x.shape[-1])
        bias_flat = torch.cat(final_biases, dim=0)
        out = _FusedMaskEstimatorFinal.apply(
            group_x,
            bias_flat,
            dim_offsets,
            extension,
            tuple(int(dim) for dim in self.dim_inputs),
            mode,
            *final_weights,
        )
        return out.reshape(*leading_shape, self._dim_total)

    forward_layer_grouped_mlp_mask_final_fused._pymsst_mask_estimator_final_fused = True
    forward_layer_grouped_mlp_mask_final_fused._pymsst_original_forward_layer_grouped_mlp = (
        original_forward_layer_grouped_mlp
    )
    bands.MaskEstimator._forward_layer_grouped_mlp = forward_layer_grouped_mlp_mask_final_fused
    return True


def _mask_estimator_final_dim_offsets(module: torch.nn.Module, device: torch.device) -> torch.Tensor:
    cache = getattr(module, "_pymsst_mask_final_offsets_cache", None)
    if cache is None:
        module._pymsst_mask_final_offsets_cache = cache = {}
    key = (device.type, device.index)
    cached = cache.get(key)
    if cached is None or cached.device != device:
        cached = torch.as_tensor(module._dim_offsets, device=device, dtype=torch.long)
        cache[key] = cached
    return cached


def _band_split_group_param_refs(module: torch.nn.Module, start: int, end: int):
    cache = getattr(module, "_pymsst_train_param_refs_cache", None)
    if cache is None:
        cache = {}
        module._pymsst_train_param_refs_cache = cache
    key = ("band_split", start, end)
    cached = cache.get(key)
    if cached is not None:
        return cached

    norms = tuple(module.to_features[i][0] for i in range(start, end))
    linears = tuple(module.to_features[i][1] for i in range(start, end))
    gammas = tuple(norm.gamma for norm in norms)
    weights = tuple(linear.weight for linear in linears)
    biases = tuple(linear.bias for linear in linears) if linears[0].bias is not None else None
    cached = (gammas, weights, biases)
    cache[key] = cached
    return cached


def _mask_layer_param_refs(module: torch.nn.Module, layer_index: int, indices: tuple[int, ...]):
    cache = getattr(module, "_pymsst_train_param_refs_cache", None)
    if cache is None:
        cache = {}
        module._pymsst_train_param_refs_cache = cache
    key = ("mask_layer", layer_index, indices)
    cached = cache.get(key)
    if cached is not None:
        return cached

    band_layers = module._band_groupable_layers()
    linears = tuple(band_layers[i][layer_index][1] for i in indices)
    weights = tuple(linear.weight for linear in linears)
    biases = tuple(linear.bias for linear in linears) if linears[0].bias is not None else None
    cached = (weights, biases)
    cache[key] = cached
    return cached


def _packed_mask_layer_param_refs(
    module: torch.nn.Module,
    estimators: tuple[torch.nn.Module, ...],
    layer_index: int,
    indices: tuple[int, ...],
):
    cache = getattr(module, "_pymsst_train_param_refs_cache", None)
    if cache is None:
        cache = {}
        module._pymsst_train_param_refs_cache = cache
    key = ("packed_mask_layer", tuple(id(estimator) for estimator in estimators), layer_index, indices)
    cached = cache.get(key)
    if cached is not None:
        return cached

    linears = tuple(
        estimator._band_groupable_layers()[band_index][layer_index][1]
        for estimator in estimators
        for band_index in indices
    )
    weights = tuple(linear.weight for linear in linears)
    biases = tuple(linear.bias for linear in linears) if linears[0].bias is not None else None
    cached = (weights, biases)
    cache[key] = cached
    return cached


def _contiguous_index_range(indices: tuple[int, ...]) -> tuple[int, int] | None:
    if not indices:
        return None
    start = int(indices[0])
    end = int(indices[-1]) + 1
    return (start, end) if indices == tuple(range(start, end)) else None


def _select_band_group(x: torch.Tensor, indices: tuple[int, ...], device: torch.device, index_tensor_fn) -> torch.Tensor:
    contiguous = _contiguous_index_range(indices)
    if contiguous is not None:
        start, end = contiguous
        return x[..., start:end, :]
    return x.index_select(-2, index_tensor_fn(indices, device))


def _copy_band_group(
    destination: torch.Tensor,
    indices: tuple[int, ...],
    source: torch.Tensor,
    device: torch.device,
    index_tensor_fn,
) -> None:
    contiguous = _contiguous_index_range(indices)
    if contiguous is not None:
        start, end = contiguous
        destination[..., start:end, :] = source
        return
    destination.index_copy_(-2, index_tensor_fn(indices, device), source)


def _stack_converted_tensors(tensors: list[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0].to(device=device, dtype=dtype).unsqueeze(0)
    if _experimental_train_param_pack_autograd_enabled() and any(tensor.requires_grad for tensor in tensors):
        return _StackConvertedTensors.apply(device, dtype, *tensors)
    return torch.stack([tensor.to(device=device, dtype=dtype) for tensor in tensors], dim=0)


def _patch_core_rotary(extension: Any) -> bool:
    if not hasattr(extension, "rotary_forward") or not hasattr(extension, "rotary_backward"):
        return False
    try:
        import pymss_core.modules.bs_roformer.transformer as transformer
    except Exception:
        return False
    if getattr(transformer.apply_rotary_emb_fast, "_pymsst_fused_rotary", False):
        return True

    original_apply_rotary = transformer.apply_rotary_emb_fast

    def apply_rotary_cuda_ext(cos: torch.Tensor, sin: torch.Tensor, t: torch.Tensor):
        if t.device.type != "cuda" or t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return original_apply_rotary(cos, sin, t)
        if t.ndim != 4 or cos.device != t.device or sin.device != t.device or cos.dtype != t.dtype or sin.dtype != t.dtype:
            return original_apply_rotary(cos, sin, t)
        return _FusedRotary.apply(t, cos[..., ::2], sin[..., ::2], extension)

    apply_rotary_cuda_ext._pymsst_fused_rotary = True
    apply_rotary_cuda_ext._pymsst_original_apply_rotary = original_apply_rotary
    transformer.apply_rotary_emb_fast = apply_rotary_cuda_ext
    return True


def _patch_core_rms_norm(extension: Any, mode: str, *, use_bwd_v2: bool = False) -> bool:
    if mode == "mixed":
        required = ("rms_norm_forward_mixed", "rms_norm_backward_mixed")
    elif mode == "fast":
        required = (
            ("rms_norm_forward_fp32_gamma", "rms_norm_backward_fp32_gamma")
            if use_bwd_v2
            else ("rms_norm_forward", "rms_norm_backward")
        )
    else:
        return False

    if any(not hasattr(extension, name) for name in required):
        return False
    try:
        import pymss_core.modules.bs_roformer.transformer as transformer
    except Exception:
        return False
    if getattr(transformer.RMSNorm.forward, "_pymsst_fused_rms_norm", False):
        return True

    original_forward = transformer.RMSNorm.forward

    def rms_norm_cuda_ext(self, x: torch.Tensor):
        if not self.training or x.device.type != "cuda" or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return original_forward(self, x)
        if use_bwd_v2 and mode == "fast" and self.gamma.dtype == torch.float32:
            return _FusedRMSNormFp32Gamma.apply(x, self.gamma, extension)
        if mode == "mixed" and self.gamma.dtype == torch.float32:
            return _FusedRMSNormMixed.apply(x, self.gamma, extension)
        gamma = self.gamma.to(device=x.device, dtype=x.dtype)
        return _FusedRMSNorm.apply(x, gamma, extension)

    rms_norm_cuda_ext._pymsst_fused_rms_norm = True
    rms_norm_cuda_ext._pymsst_fused_rms_norm_mode = mode
    rms_norm_cuda_ext._pymsst_fused_rms_norm_bwd_v2 = use_bwd_v2
    rms_norm_cuda_ext._pymsst_original_forward = original_forward
    transformer.RMSNorm.forward = rms_norm_cuda_ext
    return True


def _patch_core_attention_gates(extension: Any, mode: str) -> bool:
    if mode == "full":
        required = ("attention_gate_forward", "attention_gate_backward")
    else:
        return False

    if any(not hasattr(extension, name) for name in required):
        return False
    try:
        import pymss_core.modules.bs_roformer.transformer as transformer
    except Exception:
        return False
    if getattr(transformer.Attention.forward, "_pymsst_fused_attention_gates", False):
        return True

    original_forward = transformer.Attention.forward

    def attention_forward_cuda_ext(self, x: torch.Tensor):
        if (
            not self.training
            or x.device.type != "cuda"
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or self.to_gates.bias is None
        ):
            return original_forward(self, x)

        x = self.norm(x)
        q, k, v = transformer.qkv_to_bnhd(self.to_qkv(x), self.heads)

        if self.rotary_embed is not None:
            q, k = transformer.rotate_qk_fast_bnhd(self.rotary_embed, q, k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = self._attention(q, k, v).transpose(1, 2)
        gate_dtype = out.dtype
        gated = _FusedAttentionGate.apply(
            out,
            x.to(dtype=gate_dtype),
            self.to_gates.weight.to(dtype=gate_dtype),
            self.to_gates.bias.to(dtype=gate_dtype),
            extension,
        )
        return self.to_out(gated)

    attention_forward_cuda_ext._pymsst_fused_attention_gates = True
    attention_forward_cuda_ext._pymsst_fused_attention_gate_mode = mode
    attention_forward_cuda_ext._pymsst_original_forward = original_forward
    transformer.Attention.forward = attention_forward_cuda_ext
    return True


def _forward_grouped_band_split_cuda_ext(self, x: torch.Tensor) -> torch.Tensor:
    if not self.training or x.device.type != "cuda":
        return self._pymsst_original_forward_grouped(x)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return self._pymsst_original_forward_grouped(x)

    ext = self._pymsst_fused_extension
    bwd_mode = getattr(self, "_pymsst_fused_band_split_bwd_mode", None)
    if _experimental_band_split_ragged_forward_enabled(ext):
        leading_shape = x.shape[:-1]
        weights = []
        biases = []
        gammas = []
        for band_index, dim_in in enumerate(self.dim_inputs):
            gamma, weight, bias = self._get_group_params(band_index, band_index + 1, x.device, x.dtype)
            gammas.append(gamma[0])
            weights.append(weight[0])
            if bias is None:
                return self._pymsst_original_forward_grouped(x)
            biases.append(bias[0])
        dim_offsets = _band_split_dim_offsets(self, x.device)
        return _FusedBandSplitRagged.apply(
            x.reshape(-1, x.shape[-1]),
            dim_offsets,
            ext,
            tuple(int(dim) for dim in self.dim_inputs),
            bwd_mode,
            *gammas,
            *weights,
            *biases,
        ).reshape(*leading_shape, len(self.dim_inputs), -1)

    leading_shape = x.shape[:-1]
    outs = []
    for start, end, dim_in in self._dim_groups:
        offset_start = self._dim_offsets[start]
        offset_end = self._dim_offsets[end]
        group_x = x[..., offset_start:offset_end].reshape(-1, end - start, dim_in)
        gamma, weight, bias = self._get_group_params(start, end, x.device, x.dtype)
        outs.append(
            _FusedBandSplitGroup.apply(group_x, gamma, weight, bias, ext, bwd_mode).reshape(
                *leading_shape,
                end - start,
                -1,
            )
        )
    return torch.cat(outs, dim=-2)


def _band_split_dim_offsets(module: torch.nn.Module, device: torch.device) -> torch.Tensor:
    cache = getattr(module, "_pymsst_band_split_offsets_cache", None)
    if cache is None:
        module._pymsst_band_split_offsets_cache = cache = {}
    key = (device.type, device.index)
    cached = cache.get(key)
    if cached is None or cached.device != device:
        cached = torch.as_tensor(module._dim_offsets, device=device, dtype=torch.long)
        cache[key] = cached
    return cached


class _FusedBandSplitGroup(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gamma: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        extension,
        bwd_mode: str | None,
    ):
        out, normed, inv_norm = extension.band_split_forward(x, gamma, weight, bias)
        ctx.save_for_backward(normed, inv_norm, gamma, weight)
        ctx.has_bias = bias is not None
        ctx.extension = extension
        ctx.bwd_mode = bwd_mode
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        normed, inv_norm, gamma, weight = ctx.saved_tensors
        grad_out = grad_out.reshape(normed.shape[0], normed.shape[1], weight.shape[1]).contiguous()
        if ctx.bwd_mode == "v3":
            backward = ctx.extension.band_split_backward_v3
        elif ctx.bwd_mode == "v2":
            backward = ctx.extension.band_split_backward_v2
        else:
            backward = ctx.extension.band_split_backward
        grad_x, grad_gamma, grad_weight, grad_bias = backward(
            grad_out,
            normed,
            inv_norm,
            gamma,
            weight,
            ctx.has_bias,
        )
        if not ctx.has_bias:
            grad_bias = None
        return grad_x, grad_gamma, grad_weight, grad_bias, None, None


class _FusedBandSplitRagged(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        dim_offsets: torch.Tensor,
        extension,
        dim_inputs: tuple[int, ...],
        bwd_mode: str | None,
        *params: torch.Tensor,
    ):
        band_count = len(dim_inputs)
        gammas = list(params[:band_count])
        weights = list(params[band_count : band_count * 2])
        biases = list(params[band_count * 2 :])
        out, normed_flat, inv_norm = extension.band_split_ragged_forward(
            x,
            list(gammas),
            list(weights),
            list(biases),
            list(dim_inputs),
            dim_offsets,
        )
        ctx.extension = extension
        ctx.dim_inputs = dim_inputs
        ctx.bwd_mode = bwd_mode
        ctx.band_count = band_count
        ctx.input_shape = tuple(x.shape)
        ctx.use_ragged_backward = _experimental_band_split_ragged_backward_enabled(extension)
        ctx.save_for_backward(normed_flat, inv_norm, dim_offsets, *gammas, *weights)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        saved = ctx.saved_tensors
        normed_flat, inv_norm, dim_offsets, *params = saved
        band_count = ctx.band_count
        gammas = list(params[:band_count])
        weights = list(params[band_count:])
        grad_out = grad_out.reshape(grad_out.shape[0], band_count, weights[0].shape[0]).contiguous()
        if ctx.use_ragged_backward:
            result = ctx.extension.band_split_ragged_backward(
                grad_out,
                normed_flat,
                inv_norm,
                list(gammas),
                list(weights),
                list(ctx.dim_inputs),
                dim_offsets,
            )
            grad_x_flat = result[0]
            grad_gammas = tuple(result[1 : 1 + band_count])
            grad_weights = tuple(result[1 + band_count : 1 + band_count * 2])
            grad_biases = tuple(result[1 + band_count * 2 :])
            return (
                grad_x_flat,
                None,
                None,
                None,
                None,
                *grad_gammas,
                *grad_weights,
                *grad_biases,
            )

        grad_x_flat = grad_out.new_zeros(ctx.input_shape)
        grad_gammas = []
        grad_weights = []
        grad_biases = []
        for band_index, dim_in in enumerate(ctx.dim_inputs):
            start = int(dim_offsets[band_index].item())
            end = int(dim_offsets[band_index + 1].item())
            normed = normed_flat[:, start:end].reshape(normed_flat.shape[0], 1, dim_in)
            inv = inv_norm[:, band_index : band_index + 1]
            weight = weights[band_index].unsqueeze(0)
            gamma = gammas[band_index].unsqueeze(0)
            grad_band = grad_out[:, band_index : band_index + 1, :]
            if ctx.bwd_mode == "v3":
                backward = ctx.extension.band_split_backward_v3
            elif ctx.bwd_mode == "v2":
                backward = ctx.extension.band_split_backward_v2
            else:
                backward = ctx.extension.band_split_backward
            grad_x, grad_gamma, grad_weight, grad_bias = backward(grad_band, normed, inv, gamma, weight, True)
            grad_x_flat[:, start:end] = grad_x.reshape(grad_x.shape[0], dim_in)
            grad_gammas.append(grad_gamma[0])
            grad_weights.append(grad_weight[0])
            grad_biases.append(grad_bias[0])
        return (
            grad_x_flat,
            None,
            None,
            None,
            None,
            *grad_gammas,
            *grad_weights,
            *grad_biases,
        )


class _FusedGroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, extension):
        group_count = x.shape[-2]
        in_features = x.shape[-1]
        x_3d = x.reshape(-1, group_count, in_features)
        out = extension.grouped_linear_forward(x_3d, weight, bias)
        ctx.input_shape = tuple(x.shape)
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x_3d, weight)
        return out.reshape(*x.shape[:-2], group_count, weight.shape[1])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_3d, weight = ctx.saved_tensors
        grad_out_3d = grad_out.reshape(x_3d.shape[0], x_3d.shape[1], weight.shape[1])
        grad_x = torch.einsum("mgo,goi->mgi", grad_out_3d, weight).reshape(ctx.input_shape)
        grad_weight = torch.einsum("mgo,mgi->goi", grad_out_3d, x_3d)
        grad_bias = grad_out_3d.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_weight, grad_bias, None


class _MaskFinalGroupedLinearGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        group_count, out_features, in_features = weight.shape
        leading_shape = tuple(x.shape[:-2])
        x_3d = x.reshape(-1, group_count, in_features)
        x_t = x_3d.transpose(0, 1)
        weight_t = weight.transpose(1, 2)
        if bias is None:
            pre_t = torch.bmm(x_t, weight_t)
        else:
            pre_t = torch.baddbmm(bias.unsqueeze(1).expand(-1, x_t.shape[1], -1), x_t, weight_t)
        pre = pre_t.transpose(0, 1)
        left, gate_pre = pre.chunk(2, dim=-1)
        gate = gate_pre.sigmoid()
        ctx.input_shape = tuple(x.shape)
        ctx.leading_shape = leading_shape
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x_3d, weight, left, gate)
        return (left * gate).reshape(*leading_shape, group_count, out_features // 2)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_3d, weight, left, gate = ctx.saved_tensors
        grad = grad_out.reshape(x_3d.shape[0], x_3d.shape[1], weight.shape[1] // 2)
        grad_left = grad * gate
        grad_gate = grad * left * gate * (1 - gate)
        grad_pre = torch.cat((grad_left, grad_gate), dim=-1)
        grad_pre_for_x = grad_pre.to(dtype=weight.dtype).transpose(0, 1)
        grad_x = torch.bmm(grad_pre_for_x, weight).transpose(0, 1).reshape(ctx.input_shape)
        grad_pre_for_weight = grad_pre.to(dtype=x_3d.dtype).transpose(0, 1).transpose(1, 2)
        grad_weight = torch.bmm(grad_pre_for_weight, x_3d.transpose(0, 1))
        grad_bias = grad_pre.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_weight, grad_bias


class _FusedMaskEstimatorFinal(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        bias: torch.Tensor,
        dim_offsets: torch.Tensor,
        extension,
        dim_inputs: tuple[int, ...],
        mode: str,
        *weights: torch.Tensor,
    ):
        forward = extension.mask_estimator_final_direct_forward if mode == "direct" else extension.mask_estimator_final_forward
        out, pre, x_contiguous = forward(
            x,
            list(weights),
            bias,
            list(dim_inputs),
            dim_offsets,
        )
        ctx.extension = extension
        ctx.dim_inputs = dim_inputs
        ctx.input_shape = tuple(x.shape)
        ctx.mode = mode
        ctx.save_for_backward(x_contiguous, pre, dim_offsets, *weights)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, pre, dim_offsets, *weights = ctx.saved_tensors
        backward = (
            ctx.extension.mask_estimator_final_direct_backward
            if ctx.mode == "direct"
            else ctx.extension.mask_estimator_final_backward
        )
        result = backward(
            grad_out.reshape(grad_out.shape[0], -1).contiguous(),
            x,
            pre,
            list(weights),
            list(ctx.dim_inputs),
            dim_offsets,
        )
        grad_x = result[0].reshape(ctx.input_shape)
        grad_bias = result[1]
        grad_weights = tuple(result[2:])
        return (grad_x, grad_bias, None, None, None, None, *grad_weights)


class _StackConvertedTensors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, device: torch.device, dtype: torch.dtype, *tensors: torch.Tensor):
        ctx.input_meta = tuple((tensor.device, tensor.dtype) for tensor in tensors)
        ctx.input_count = len(tensors)
        return torch.stack([tensor.to(device=device, dtype=dtype) for tensor in tensors], dim=0)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grads = []
        for index, (device, dtype) in enumerate(ctx.input_meta):
            grad = grad_out[index]
            if grad.device != device or grad.dtype != dtype:
                grad = grad.to(device=device, dtype=dtype)
            grads.append(grad)
        return None, None, *grads


class _FusedRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, extension):
        out = extension.rotary_forward(t, cos, sin)
        ctx.save_for_backward(cos.contiguous(), sin.contiguous())
        ctx.extension = extension
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        cos, sin = ctx.saved_tensors
        grad_t = ctx.extension.rotary_backward(grad_out, cos, sin)
        return grad_t, None, None, None


class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, extension):
        out, normed, inv_norm = extension.rms_norm_forward(x, gamma)
        ctx.input_shape = tuple(x.shape)
        ctx.save_for_backward(normed, inv_norm, gamma)
        ctx.extension = extension
        return out.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        normed, inv_norm, gamma = ctx.saved_tensors
        grad_x, grad_gamma = ctx.extension.rms_norm_backward(grad_out.reshape(ctx.input_shape), normed, inv_norm, gamma)
        return grad_x.reshape(ctx.input_shape), grad_gamma, None


class _FusedRMSNormFp32Gamma(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, extension):
        out, normed, inv_norm = extension.rms_norm_forward_fp32_gamma(x, gamma)
        ctx.input_shape = tuple(x.shape)
        ctx.save_for_backward(normed, inv_norm, gamma)
        ctx.extension = extension
        return out.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        normed, inv_norm, gamma = ctx.saved_tensors
        grad_x, grad_gamma = ctx.extension.rms_norm_backward_fp32_gamma(
            grad_out.reshape(ctx.input_shape),
            normed,
            inv_norm,
            gamma,
        )
        return grad_x.reshape(ctx.input_shape), grad_gamma, None


class _FusedRMSNormMixed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, extension):
        out, normed, inv_norm = extension.rms_norm_forward_mixed(x, gamma)
        ctx.input_shape = tuple(x.shape)
        ctx.save_for_backward(normed, inv_norm, gamma, x.new_empty(0))
        ctx.extension = extension
        return out.reshape(*x.shape[:-1], x.shape[-1])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        normed, inv_norm, gamma, x_meta = ctx.saved_tensors
        grad_x, grad_gamma = ctx.extension.rms_norm_backward_mixed(
            grad_out.reshape(ctx.input_shape),
            normed,
            inv_norm,
            gamma,
            x_meta,
        )
        return grad_x.reshape(ctx.input_shape), grad_gamma, None


class _FusedAttentionGate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        attn: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        extension,
    ):
        attn_shape = tuple(attn.shape)
        x_shape = tuple(x.shape)
        heads = attn.shape[-2]
        dim_head = attn.shape[-1]
        attn_3d = attn.reshape(-1, heads, dim_head)
        x_2d = x.reshape(-1, x.shape[-1])
        out, gate, saved_attn, saved_x = extension.attention_gate_forward(attn_3d, x_2d, weight, bias)
        ctx.attn_shape = attn_shape
        ctx.x_shape = x_shape
        ctx.save_for_backward(saved_attn, saved_x, weight, gate)
        ctx.extension = extension
        return out.reshape(*x.shape[:-1], heads * dim_head)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        saved_attn, saved_x, weight, gate = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(saved_attn.shape[0], saved_attn.shape[1] * saved_attn.shape[2])
        grad_attn, grad_x, grad_weight, grad_bias = ctx.extension.attention_gate_backward(
            grad_out_2d,
            saved_attn,
            saved_x,
            weight,
            gate,
        )
        return grad_attn.reshape(ctx.attn_shape), grad_x.reshape(ctx.x_shape), grad_weight, grad_bias, None


def reference_band_split_group(x: torch.Tensor, gamma: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    dim_in = x.shape[-1]
    normalized = F.normalize(x, dim=-1) * (dim_in**0.5) * gamma.unsqueeze(0)
    out = torch.einsum("mgi,goi->mgo", normalized, weight)
    if bias is not None:
        out = out + bias.unsqueeze(0)
    return out
