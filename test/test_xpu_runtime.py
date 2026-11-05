from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from pymss import separator as separator_module
from test.test_separator_device_defaults import DummyLogger


def _available_xpu(monkeypatch, count=3):
    monkeypatch.setattr(separator_module, "_xpu_runtime_available", lambda: True)
    monkeypatch.setattr(torch, "xpu", SimpleNamespace(device_count=lambda: count), raising=False)


@pytest.mark.parametrize("device", ["xpu", "auto"])
@pytest.mark.parametrize("ids", [[0], [1], [2, 0]])
def test_xpu_selection_uses_the_requested_first_device(monkeypatch, device, ids):
    _available_xpu(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert separator_module._select_device(device, ids, DummyLogger()) == f"xpu:{ids[0]}"


@pytest.mark.parametrize("device", ["xpu", "auto"])
@pytest.mark.parametrize("ids", [None, [], [-1], [3], [0, 3], [True], [1.5], ["1"]])
def test_invalid_xpu_ids_fail_before_loading_a_model(monkeypatch, device, ids):
    _available_xpu(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="XPU"):
        separator_module._select_device(device, ids, DummyLogger())


def test_auto_prefers_xpu_over_npu(monkeypatch):
    _available_xpu(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(separator_module, "_npu_runtime_available", lambda: True)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device_count=lambda: 1), raising=False)
    assert separator_module._select_device("auto", [0], DummyLogger()) == "xpu:0"


def test_auto_prefers_cuda_without_initializing_xpu(monkeypatch):
    def forbidden_probe():
        raise AssertionError("CUDA selection must not initialize XPU")

    monkeypatch.setattr(separator_module, "_xpu_runtime_available", forbidden_probe)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert separator_module._select_device("auto", [0], DummyLogger()) == "cuda:0"


def test_explicit_xpu_fails_when_runtime_is_unavailable(monkeypatch):
    monkeypatch.setattr(separator_module, "_xpu_runtime_available", lambda: False)
    with pytest.raises(RuntimeError, match="XPU-enabled"):
        separator_module._resolve_public_device("xpu", {}, DummyLogger())


@pytest.mark.parametrize(
    "argv",
    [
        ["infer", "model", "-i", "input.wav", "--device", "xpu"],
        ["workflow", "run", "-c", "workflow.yaml", "-i", "input.wav", "--device", "xpu"],
        ["comfy", "run", "-c", "workflow.json", "--device", "xpu"],
        ["serve", "--device", "xpu", "--device-id", "1"],
    ],
)
def test_cli_accepts_xpu(argv):
    from pymss.cli import build_parser

    assert build_parser().parse_args(argv).device == "xpu"


@pytest.mark.parametrize("device", ["xpu", "xpu:0"])
@pytest.mark.parametrize("enabled", [True, False])
def test_xpu_autocast_explicitly_uses_fp32(monkeypatch, device, enabled):
    from pymss.utils import _autocast

    autocast = Mock(side_effect=AssertionError("XPU must not enable AMP"))
    monkeypatch.setattr(torch.amp, "autocast", autocast)
    assert isinstance(_autocast(device, enabled), nullcontext)
    autocast.assert_not_called()


@pytest.mark.parametrize("device", ["xpu:0", "cpu"])
def test_del_cache_only_clears_xpu_for_xpu_device(monkeypatch, device):
    empty_cache = Mock()
    collect = Mock()
    monkeypatch.setattr(torch, "xpu", SimpleNamespace(empty_cache=empty_cache), raising=False)
    monkeypatch.setattr(separator_module.gc, "collect", collect)
    separator = SimpleNamespace(device=device, logger=DummyLogger())

    separator_module.MSSeparator.del_cache(separator)

    collect.assert_called_once_with()
    assert empty_cache.call_count == (1 if device.startswith("xpu") else 0)
