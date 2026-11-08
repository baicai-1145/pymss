"""MUSA (Moore Threads) profile contract.

The flags below are not guesses: every one of them was measured on MTT S70 +
torch 2.5.0 + torch_musa 2.5.0 + MUSA SDK 4.2 with ``tools/musa_probe.py``
(see ``docs/reports/musa-s70-probe-findings.md``).  This test keeps them from
drifting silently when a vendor stack is upgraded.

Runs on any host: ``device.py`` resolves an unregistered device type from its
prefix, so no MUSA hardware or ``torch_musa`` package is needed here.
"""
from __future__ import annotations

import torch

from pymss.backends import (
    SUPPORTED_DEVICES,
    amp_supported,
    fft_fallback_required,
    rnn_fallback_required,
)
from pymss_core.device import clear_device_cache, profile_for_alias


def setup_function():
    clear_device_cache()


def teardown_function():
    clear_device_cache()


def test_musa_aliases_are_selectable():
    for alias in ("musa", "moore"):
        assert alias in SUPPORTED_DEVICES


def test_moore_profile_matches_measured_flags():
    profile = profile_for_alias("musa")
    assert (profile.vendor, profile.device_type, profile.plugin) == ("moore", "musa", "torch_musa")
    # torch.istft rejects the complex tensor -> composed torch.fft path.
    assert profile.complex_fft is False
    # muFFT has no half transforms below SM_53 and S70 reports SM_21.
    assert profile.fft_dtype_shim is True
    # irfft on a transposed complex tensor faults (and poisons the context).
    assert profile.fft_shim is True
    # No complex-by-real pointwise kernel: scale/compose on the real planes.
    assert profile.complex_mul is False
    assert profile.complex_reshape is False
    # torch.cat on complex tensors raises SetMUTensorDType.
    assert profile.complex_cat is True
    # Complex scatter_add_/zero-fill are missing: fold on the real planes.
    assert profile.scatter_add is False
    # nn.MultiheadAttention's fused kernel (_native_multi_head_attention) is CUDA-only.
    assert profile.sdpa_shim is True
    # LSTM/RNN are numerically correct; only GRU lacks _thnn_fused_gru_cell.
    assert profile.rnn is True
    assert profile.rnn_blocked == frozenset({"gru"})
    # fp16 is unusable for real models even though the micro benchmarks pass:
    # MUDNN's fused fp16 conv aborts (scnet/mdx23c/htdemucs) and fp16 complex
    # does not exist (mel_band/bandit).  bfloat16 is worse.
    assert profile.amp is False
    assert profile.amp_dtype == torch.float16
    assert profile.multi_device is False


def test_musa_fallbacks_route_as_measured():
    assert fft_fallback_required("musa:0") is True
    assert rnn_fallback_required("musa:0") is False   # native LSTM must not bounce
    assert amp_supported("musa:0") is False   # fp16 breaks real models
