"""MUSA (Moore Threads) profile contract.

The flags below are not guesses: every one of them was measured on MTT S70 +
torch 2.7.1 + torch_musa 2.7.1 + MUSA SDK 4.3.0 + driver 3.3.0 with
``tools/musa_probe.py`` (see ``docs/reports/musa-2.7.1-upgrade-results.md``).
This test keeps them from drifting silently when a vendor stack is upgraded.

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
    # Native STFT/ISTFT work end to end now; the composed path matches CPU to 2.6e-07.
    assert profile.complex_fft is True
    # muFFT has no half transforms below mp 2.2 and the S70 reports 2.1.
    assert profile.fft_dtype_shim is True
    # Native rfft/irfft are numerically correct on 2.7 (1.9e-07), so no exact DFT.
    assert profile.fft_shim is False
    # complex-by-real multiply/divide, complex cat and complex fill all exist now.
    assert profile.complex_mul is True
    assert profile.complex_reshape is True
    assert profile.complex_cat is False
    # Complex scatter_add_/index_add_ are still missing: fold on the real planes.
    assert profile.scatter_add is False
    # The fused rope kernel is 7.6x faster than the reference rotary path and is
    # numerically verified on this card, so the shim is on.
    assert profile.rope_kernel_shim is True
    # nn.MultiheadAttention runs on the native op now (3.4e-07).
    assert profile.sdpa_shim is False
    # GRU is native (_thnn_fused_gru_cell exists), like LSTM.
    assert profile.rnn is True
    assert profile.rnn_blocked == frozenset()
    # fp16 still aborts on real models: MUDNN's fused fp16 conv fails on
    # conv1d/conv2d and bfloat16 fails in datacast, so inference stays fp32.
    assert profile.amp is False
    assert profile.amp_dtype == torch.float16
    assert profile.multi_device is False


def test_musa_fallbacks_route_as_measured():
    assert fft_fallback_required("musa:0") is False   # native FFT is correct
    assert rnn_fallback_required("musa:0") is False   # native LSTM/GRU must not bounce
    assert amp_supported("musa:0") is False           # fp16 breaks real models
