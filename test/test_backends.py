from __future__ import annotations

import pytest
import torch

from pymss.backends import (
    SUPPORTED_DEVICES,
    amp_supported,
    device_profile,
    install_conv_fallback,
    install_recurrent_fallback,
    prepare_device_alias,
)
from pymss_core.device import clear_device_cache


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    clear_device_cache()
    for name in ("PYMSS_DEVICE_BACKEND", "PYMSS_FFT_FALLBACK", "PYMSS_RNN_FALLBACK", "PYMSS_AMP"):
        monkeypatch.delenv(name, raising=False)
    yield
    clear_device_cache()


def test_supported_devices_covers_every_alias():
    for alias in ("auto", "cpu", "cuda", "rocm", "mps", "mlx", "npu", "gcu", "mlu", "supa", "biren", "iluvatar", "metax", "enflame", "hygon"):
        assert alias in SUPPORTED_DEVICES


def test_cli_device_choices_match_backends():
    from pymss.cli import build_parser

    actions = [action for action in _walk_actions(build_parser()) if action.dest == "device"]
    assert actions, "no --device argument found in the CLI parser"
    for action in actions:
        assert set(SUPPORTED_DEVICES) <= set(action.choices)


def _walk_actions(parser):
    import argparse

    for action in parser._actions:
        yield action
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield from _walk_actions(subparser)


def test_install_recurrent_fallback_is_noop_on_cpu():
    assert install_recurrent_fallback("cpu") is False


def test_install_recurrent_fallback_patches_on_ascend():
    assert install_recurrent_fallback("npu:0") is True
    # Idempotent: a second call must not double-wrap.
    assert install_recurrent_fallback("npu:0") is True


def test_amp_not_reported_for_cpu():
    assert amp_supported("cpu") is False


def test_domestic_vendor_alias_resolves_to_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "Iluvatar BI-V150", raising=False)
    assert prepare_device_alias("iluvatar") == "cuda"
    assert device_profile("cuda").vendor == "iluvatar"


def test_install_conv_fallback_is_noop_where_dilation_is_supported():
    assert install_conv_fallback("cpu") is False
    assert install_conv_fallback("cuda:0") is False


def test_install_conv_fallback_patches_on_biren():
    """BIREN's SUDNN aborts the process on dilated convs, so the shim must load."""
    assert install_conv_fallback("supa:0") is True
    assert install_conv_fallback("supa:0") is True
