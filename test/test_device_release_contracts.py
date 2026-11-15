from contextlib import nullcontext
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pymss import separator as module
from pymss.config import AttrDict
from pymss.utils import _autocast, _resolve_use_amp
from pymss_core.device import clear_device_cache


@pytest.fixture(autouse=True)
def clear_profiles():
    clear_device_cache()
    yield
    clear_device_cache()


@pytest.mark.parametrize('ids', [None, [], [-1], [True], [1.5], ['1'], [1]])
def test_auto_private_backend_validates_device_ids(monkeypatch, ids):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(module, '_xpu_runtime_available', lambda: False)
    monkeypatch.setattr(module, '_npu_runtime_available', lambda: False)
    monkeypatch.setattr(module, 'prepare_device_alias', lambda alias: 'mlu')
    monkeypatch.setattr(torch, 'mlu', SimpleNamespace(is_available=lambda: True, device_count=lambda: 1), raising=False)
    with pytest.raises(ValueError, match='MLU'):
        module._select_device('auto', ids, logging.getLogger(__name__))


def test_explicit_private_backend_without_devices_fails(monkeypatch):
    monkeypatch.setattr(torch, 'musa', SimpleNamespace(is_available=lambda: False, device_count=lambda: 0), raising=False)
    with pytest.raises(RuntimeError, match='[Mm][Oo][Oo][Rr][Ee]|MUSA|musa'):
        module._select_device('musa', [0], logging.getLogger(__name__))


def test_auto_musa_keeps_selected_device_and_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(module, '_xpu_runtime_available', lambda: False)
    monkeypatch.setattr(module, '_npu_runtime_available', lambda: False)

    def prepare(alias):
        if alias != 'musa':
            raise ImportError(alias)
        return 'musa'

    monkeypatch.setattr(module, 'prepare_device_alias', prepare)
    monkeypatch.setattr(torch, 'musa', SimpleNamespace(is_available=lambda: True, device_count=lambda: 2), raising=False)
    assert module._select_device('auto', [1], logging.getLogger(__name__)) == 'musa:1'


@pytest.mark.parametrize('failure_point', ['prepare', 'probe'])
@pytest.mark.parametrize('error_type', [RuntimeError, OSError])
def test_auto_skips_a_broken_optional_backend(monkeypatch, failure_point, error_type, caplog):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(module, '_xpu_runtime_available', lambda: False)
    monkeypatch.setattr(module, '_npu_runtime_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)

    def fail():
        raise error_type('Backend initialization failed')

    def prepare(alias):
        if alias != 'gcu':
            raise ImportError(alias)
        if failure_point == 'prepare':
            fail()
        return 'xla'

    monkeypatch.setattr(module, 'prepare_device_alias', prepare)
    monkeypatch.setattr(torch, 'xla', SimpleNamespace(is_available=fail), raising=False)
    with caplog.at_level(logging.DEBUG):
        assert module._select_device('auto', [0], logging.getLogger(__name__)) == 'cpu'
    assert 'gcu' in caplog.text and 'Backend initialization failed' in caplog.text


def test_unsupported_private_device_never_falls_back_to_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    with pytest.raises(RuntimeError, match='device profile'):
        module._select_device('custom_backend', [0], logging.getLogger(__name__))


@pytest.mark.parametrize('mode', ['profile', 'environment', 'constructor', 'disabled_context'])
def test_effective_amp_disablement_promotes_half_checkpoint(monkeypatch, mode):
    from pymss import utils

    separator = module.MSSeparator.__new__(module.MSSeparator)
    separator.model_type = 'scnet'
    separator.model_path = 'model.ckpt'
    separator.config_path = 'model.yaml'
    separator.device = 'cuda:0'
    separator.device_ids = [0]
    separator.logger = logging.getLogger(__name__)
    separator.inference_params = {}
    separator.target_instrument_override = None
    separator._log_model_config = lambda *args, **kwargs: None
    config = AttrDict({'audio': {}, 'training': {'instruments': ['vocals'], 'use_amp': True}, 'inference': {}})
    model = torch.nn.Conv1d(2, 2, 1)
    state = torch.nn.Conv1d(2, 2, 1).half().state_dict()
    monkeypatch.setattr(module, '_load_state_dict', lambda *args: state)
    monkeypatch.setattr(module, 'get_model_from_config', lambda *args, **kwargs: (model, config))
    monkeypatch.setattr(model, 'to', lambda *args, **kwargs: model)
    if mode == 'profile':
        monkeypatch.setattr(utils, 'amp_supported', lambda device: False)
    elif mode == 'environment':
        monkeypatch.setenv('PYMSS_AMP', '0')
        clear_device_cache()
    elif mode == 'disabled_context':
        monkeypatch.setattr(utils, 'amp_supported', lambda device: True)
        monkeypatch.setattr(torch.amp, 'autocast', lambda *args, **kwargs: SimpleNamespace(_enabled=False))
    else:
        monkeypatch.setattr(utils, 'amp_supported', lambda device: True)

        def reject(*args, **kwargs):
            raise RuntimeError('Autocast is unavailable')

        monkeypatch.setattr(torch.amp, 'autocast', reject)
    loaded, _ = separator.load_model()
    assert loaded.weight.dtype == torch.float32
    assert _resolve_use_amp(config) is True
    assert isinstance(_autocast('cuda:0', True), nullcontext)
    assert torch.isfinite(loaded(torch.randn(1, 2, 16))).all()


def test_native_autocast_that_disables_itself_is_treated_as_amp_off(monkeypatch):
    from pymss import utils

    monkeypatch.setattr(utils, 'amp_supported', lambda device: True)
    monkeypatch.setattr(utils, 'autocast_dtype', lambda device: torch.float32)
    with pytest.warns(UserWarning, match='not supported'):
        context = _autocast('cpu', True)
    assert isinstance(context, nullcontext)
    config = AttrDict({'training': {'use_amp': True}, 'inference': {}})
    with pytest.warns(UserWarning, match='not supported'):
        assert _resolve_use_amp(config, 'cpu') is False


def test_vr_uploads_complex_spectrum_as_real_planes(monkeypatch):
    from pymss.modules.vocal_remover import vr_separator

    separator = vr_separator.VRSeparator.__new__(vr_separator.VRSeparator)
    separator.model_run = SimpleNamespace(offset=0, predict_mask=lambda batch: torch.full_like(batch, .5))
    separator.mps_model_backend = 'torch'
    separator.window_size = 4
    separator.batch_size = 2
    separator.debug = False
    separator.progress_callback = None
    separator.use_channels_last = False
    separator.use_amp = False
    separator.enable_post_process = False
    separator.primary_stem_name = 'vocals'
    separator.enable_tta = False
    original = torch.Tensor.to

    def reject_complex(tensor, *args, **kwargs):
        if tensor.is_complex():
            raise RuntimeError('Complex host-to-device transfers are unavailable')
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', reject_complex)
    spectrum = np.full((2, 3, 8), 1 + 2j, dtype=np.complex64)
    primary, secondary = separator.inference_vr(spectrum, 'cpu', {'value': 0})
    np.testing.assert_allclose(primary + secondary, spectrum)
    np.testing.assert_allclose(primary, spectrum * .5)
