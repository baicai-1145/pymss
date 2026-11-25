"""CPU fallback for convolutions the vendor kernel cannot plan.

BIREN's SUDNN rejects most dilated convolutions on the current SUPA stack
(1.9.0.0.rc1 / torch-br 1.7.0.20700).  The failure is not a Python error: the
plan builder throws a C++ exception that reaches ``terminate()``, so a model
with a dilated resblock kills the whole process instead of raising.  Measured
on a BIREN106M (2x192x21x32 input, 3x3 kernel, fp32):

===================  =========
dilation / padding   result
===================  =========
(1, 1)               native OK
(2, 2) / (2, 2)      native OK
(4, 4) / (4, 4)      becomes SUDNN plan failure
(4, 2), (1, 2), (2, 1) any padding -> process abort
BRTB_ENABLE_SUPA_FALLBACK=1        no effect on conv
bfloat16             no effect
===================  =========

With ``BRTB_ENABLE_FORCE_EAGER_CONV2D=1`` (what pymss sets for BIREN, see
``device._VENDOR_ENV_DEFAULTS``) the convolution never reaches the SUDNN plan
builder and every combination above works on the device:

===================================  =========  ========
padding / dilation                   shape      max err
===================================  =========  ========
(0,1) (1,2)                          (1,8,62,62)  5.8e-05
(1,0) (1,2)                          (1,8,64,60)  6.4e-05
(2,1) (1,3)                          (1,8,66,60)  6.3e-05
(1,2) (2,1)                          (1,8,62,66)  5.9e-05
(2,2) (2,2) symmetric                (1,8,64,64)  5.5e-05
===================================  =========  ========

``dilated_conv_blocked()`` therefore returns ``False`` whenever the eager
kernels are active, and this fallback only guards the SUDNN path.

Dilated layers are small in the models that use them, so bouncing those calls
through CPU keeps inference usable.  Inference only: a call with autograd
enabled raises instead of silently detaching.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pymss_core.device import dilated_conv_blocked

_installed = False


def _dilation_of(conv) -> tuple[int, int]:
    from pymss_core.device import _normalise_dilation

    return _normalise_dilation(conv.dilation)


def needs_cpu_fallback(conv, input) -> bool:
    """Return ``True`` when this conv must run on CPU for the current device."""
    if not isinstance(input, torch.Tensor) or not hasattr(conv, "weight"):
        return False
    if input.dim() < 3:
        return False
    dilation = _dilation_of(conv)
    if dilation == (1, 1):
        return False
    return dilated_conv_blocked(input.device, dilation)


def replicate_pad_bins(x: torch.Tensor, target_bins: int, dim: int = 2) -> torch.Tensor:
    """Replicate-pad along the bin dimension without ``F.pad``.

    ``F.pad(..., mode="replicate")`` maps to ``aten::replication_pad2d``, which
    the SUPA backend reports as "not currently supported ... will fall back to
    run on the CPU" and then fails outright inside SUDNN:

        [reorder] ReorderWei2Plain() FAILED
        Kernel execution failed with suError: 101

    Repeating the last row is the definition of replicate padding for a
    trailing-only pad, and every backend supports ``cat``/``expand``.

    Args:
        x: Tensor shaped ``(batch, channels, bins, frames)`` (or ``dim`` bins).
        target_bins: Desired size of the padded dimension.
        dim: Dimension to pad.

    Returns:
        ``x`` padded on the trailing side of ``dim`` to ``target_bins``.
    """
    pad = target_bins - x.shape[dim]
    if pad <= 0:
        return x
    index = [slice(None)] * x.dim()
    index[dim] = slice(-1, None)
    edge = x[tuple(index)]
    shape = list(x.shape)
    shape[dim] = pad
    return torch.cat([x, edge.expand(*shape)], dim=dim)


def _conv_on_cpu(conv, input):
    """Evaluate ``conv`` on CPU in fp32 and move the result back."""
    if torch.is_grad_enabled():
        raise RuntimeError(
            "The dilated-conv CPU fallback supports inference only; disable autograd before calling it."
        )
    device, dtype = input.device, input.dtype
    weight = conv.weight.detach().to(device="cpu", dtype=torch.float32)
    bias = conv.bias.detach().to(device="cpu", dtype=torch.float32) if conv.bias is not None else None
    args = (input.detach().to(device="cpu", dtype=torch.float32), weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups)
    if input.dim() == 4:
        out = F.conv2d(*args)
    elif input.dim() == 5:
        out = F.conv3d(*args)
    else:
        out = F.conv1d(*args)
    return out.to(device=device, dtype=dtype)


def _functional_fallback(fn, dim: int):
    """Wrap ``torch.nn.functional.conv{1,2,3}d`` with the CPU bounce."""

    def conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if _functional_needs_cpu(input, dilation):
            return _functional_on_cpu(fn, dim, input, weight, bias, stride, padding, dilation, groups)
        return fn(input, weight, bias, stride, padding, dilation, groups)

    conv.__name__ = getattr(fn, "__name__", "conv")
    conv.__doc__ = getattr(fn, "__doc__", None)
    return conv


def _functional_needs_cpu(input, dilation) -> bool:
    if not isinstance(input, torch.Tensor):
        return False
    from pymss_core.device import _normalise_dilation, dilated_conv_blocked

    if _normalise_dilation(dilation) == (1, 1):
        return False
    return dilated_conv_blocked(input.device, dilation)


def _functional_on_cpu(fn, dim, input, weight, bias, stride, padding, dilation, groups):
    """Run the functional conv on CPU in fp32 and move the result back."""
    if torch.is_grad_enabled():
        raise RuntimeError(
            "The dilated-conv CPU fallback supports inference only; disable autograd before calling it."
        )
    device, dtype = input.device, input.dtype
    out = fn(
        input.detach().to(device="cpu", dtype=torch.float32),
        weight.detach().to(device="cpu", dtype=torch.float32),
        None if bias is None else bias.detach().to(device="cpu", dtype=torch.float32),
        stride,
        padding,
        dilation,
        groups,
    )
    return out.to(device=device, dtype=dtype)


def install_dilated_conv_fallback():
    """Patch the conv entry points so unsupported dilated convs run on CPU.

    Both ``_ConvNd.forward`` and ``torch.nn.functional.conv{1,2,3}d`` are hooked:
    some model code bypasses the module and calls the functional form directly,
    which is exactly how the first BIREN VR run still aborted after the module
    level patch was in place.
    """
    global _installed
    if _installed:
        return
    from torch.nn.modules.conv import _ConvNd

    original = getattr(_ConvNd.forward, "_pymss_conv_original_forward", _ConvNd.forward)

    def forward(self, input):
        if needs_cpu_fallback(self, input):
            return _conv_on_cpu(self, input)
        return original(self, input)

    forward._pymss_conv_original_forward = original
    _ConvNd.forward = forward

    import torch.nn.functional as F

    for name in ("conv1d", "conv2d", "conv3d"):
        original_fn = getattr(F, name)
        if getattr(original_fn, "_pymss_conv_original", False):
            continue
        wrapper = _functional_fallback(original_fn, int(name[-2]))
        wrapper._pymss_conv_original = True
        setattr(F, name, wrapper)
        if hasattr(torch, name):
            original_torch = getattr(torch, name)
            if not getattr(original_torch, "_pymss_conv_original", False):
                torch_wrapper = _functional_fallback(original_torch, int(name[-2]))
                torch_wrapper._pymss_conv_original = True
                setattr(torch, name, torch_wrapper)
    _installed = True


__all__ = ["install_dilated_conv_fallback", "needs_cpu_fallback"]
