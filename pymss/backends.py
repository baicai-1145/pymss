"""Accelerator backend helpers.

pymss runs on CUDA, ROCm, Apple MPS/MLX, Ascend NPU, and the domestic
accelerator families (Biren, Iluvatar, MetaX, Enflame, Hygon DCU, Cambricon
MLU).  Those SDKs disagree about how they surface to ``torch``: some add a
private device type (``npu``/``gcu``/``mlu``/``supa``), others impersonate CUDA.
:mod:`pymss_core.device` owns the detection matrix; this module is the thin
surface pymss itself uses so that no caller needs to hardcode ``device.type``
comparisons.

Example:
    >>> from pymss.backends import install_recurrent_fallback, amp_supported
    >>> install_recurrent_fallback("npu:0")
    True
    >>> amp_supported("cuda:0")
    True
"""
from __future__ import annotations

from pymss_core.device import (
    DEVICE_ALIASES,
    DeviceProfile,
    amp_supported,
    clear_device_cache,
    detect_vendor,
    device_profile,
    fft_fallback_required,
    prepare_device_alias,
    requires_plugin,
    rnn_fallback_required,
)

#: Values accepted by ``--device``; ``auto`` is resolved by the separator.
SUPPORTED_DEVICES: tuple[str, ...] = ("auto", *DEVICE_ALIASES)

__all__ = (
    "SUPPORTED_DEVICES",
    "DeviceProfile",
    "amp_supported",
    "clear_device_cache",
    "detect_vendor",
    "device_profile",
    "fft_fallback_required",
    "install_recurrent_fallback",
    "prepare_device_alias",
    "requires_plugin",
    "rnn_fallback_required",
)


def install_recurrent_fallback(device) -> bool:
    """Install the CPU recurrent fallback when ``device`` needs it.

    Ascend NPU builds ship inference-incomplete ``npu_lstm``/``npu_gru``
    kernels, and any future accelerator whose profile reports ``rnn=False``
    hits the same wall.  Native backends are left untouched.

    Args:
        device: Device string, ``torch.device`` or tensor.

    Returns:
        ``True`` when the fallback was installed.
    """
    if not rnn_fallback_required(device):
        return False
    from pymss_core.npu_rnn import install_rnn_cpu_fallback

    install_rnn_cpu_fallback()
    return True
