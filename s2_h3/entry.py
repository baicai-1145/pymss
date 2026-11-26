"""ARM entry for BS-RoFormer temporal-branch experiments (Colab).  Same protocol as the archived conv62s/s_aw runs.

ARMS (temporal branch applied per stack before the frequency transformer):
  A       stock BS-RoFormer (time attention + FFN)                       24,591,340 params
  S       RMSNorm -> u*SiLU(v) -> DWConv(k=31)          -> Linear        21,852,956
  S2lite  S with parallel dilated DWConv {1, 8} + gate dropout            21,926,684
  S2      S with parallel dilated DWConv {1, 4, 16} + gate dropout        22,000,412
  BandUNet frequency U-Net with 62→32→16→8 learned band scales         20,961,844
  H3 / Largo
          H2 with stacks 0-2 = gated static band mixer, stacks 3-5 = slow routing (q/k every 4 frames) + narrow v
          + FFN every 4 frames (h3.py; frozen audit in .tmp/freq-attn-audit/FINDINGS.md H)   20,274,048

Names (ALIASES below; both spellings work everywhere, including ARM=<name>):  A = Baseline, H2 = PoolKV, H2M/H2MG/H2N128
= PoolKV-M / PoolKV-MG / PoolKV-N128, H3 = StaticStride.
"""
import copy
import os
import runpy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from pymss_core import get_model_from_config
from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
from pymss_core.modules.bs_roformer.common import mask_to_complex_shape
from pymss_core.modules.bs_roformer.transformer import FeedForward, RMSNorm
from pooled_kv import PooledKVAttention
from frequency_grid import BandFFN, BandMixer

ROOT = Path(__file__).resolve().parent
CFG = ROOT / 'aw.yaml'
# S2_DEVICE=npu runs the same protocol on Ascend (torch_npu); unset = CUDA where available, else CPU.
DEVICE = torch.device(os.environ.get('S2_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu'))
ACCELERATOR = {'cuda': 'gpu', 'npu': 'npu'}.get(DEVICE.type, 'cpu')
if DEVICE.type == 'npu':
    import torch_npu  # noqa: E402,F401  -- registers the npu device type
ARMS = {
    'S': dict(dilations=(1,), mult=4, gate_dropout=False, params=21852956),
    'S2lite': dict(dilations=(1, 8), mult=4, gate_dropout=True, params=21926684),
    'S2': dict(dilations=(1, 4, 16), mult=4, gate_dropout=True, params=22000412),
    'H2': dict(dilations=(1,), mult=4, gate_dropout=False, params=21582624),
    'H2M': dict(dilations=(1,), mult=4, gate_dropout=False, params=19398576),
    'H2MG': dict(dilations=(1,), mult=4, gate_dropout=False, params=19400112),
    'H2N128': dict(params=10440536),
    'BandUNet': dict(params=20961844),
    'H3': dict(dilations=(1,), mult=4, gate_dropout=False, params=20274048),
}

# Names: the mechanism, the way a paper would state it.  PoolKV = frequency-axis attention whose keys/values are pooled
# into the slow-moving band summary (layers 0 and 4 keep attention, the rest is unchanged).  StaticStride = layers 0-2
# drop attention for a static band mixer, layers 3-5 compute q/k every 4th frame with a narrow v and a 4-frame FFN.
# The code keeps the original keys so every archived result, weight file and run directory still resolves.
ALIASES = {
    'Baseline': 'A', 'baseline': 'A',
    'PoolKV': 'H2', 'PoolKV-M': 'H2M', 'PoolKV-MG': 'H2MG', 'PoolKV-N128': 'H2N128',
    'StaticStride': 'H3',
}


def resolve(arm):
    return ALIASES.get(arm, arm)
A_PARAMS = 24591340
H2_ATTENTION_LAYERS = (0, 4)


class MSC(nn.Module):
    """Gated multi-scale temporal conv: Linear(d->2h) -> u*SiLU(v) -> [dropout] -> sum_d DWConv1d(k, dilation d) -> Linear(h->d)."""
    def __init__(self, dim, mult=4, dilations=(1,), k=31, dropout=0.0, gate_dropout=False):
        super().__init__()
        h = dim * mult // 2
        self.proj_in = nn.Linear(dim, h * 2)
        self.convs = nn.ModuleList(nn.Conv1d(h, h, k, padding=(k // 2) * d, dilation=d, groups=h) for d in dilations)
        self.proj_out = nn.Linear(h, dim)
        self.drop = nn.Dropout(dropout)
        self.gate_drop = nn.Dropout(dropout) if gate_dropout else nn.Identity()
        self.dilations, self.k, self.mult, self.swiglu = tuple(dilations), k, mult, True

    def forward(self, x):  # (b, t, f, d)
        b, t, f, d = x.shape
        u, v = self.proj_in(x).chunk(2, dim=-1)
        g = self.gate_drop(u * F.silu(v))
        hid = g.shape[-1]
        g = g.permute(0, 2, 1, 3).reshape(b * f, t, hid).transpose(1, 2)
        g = sum(c(g) for c in self.convs)
        g = g.transpose(1, 2).reshape(b, f, t, hid).permute(0, 2, 1, 3)
        return self.drop(self.proj_out(g))


class ArmModel(BSRoformer):
    def _forward_mask_core(self, st):
        b, fs, t, ri = st.shape
        features = st.permute(0, 2, 1, 3).reshape(b, t, fs * ri).to(next(self.band_split.parameters()).dtype)
        x = self.band_split(features)
        for i, (_, freq) in enumerate(self.layers):
            x = self.convs[i](self.temporal_norms[i](x)) + x
            if getattr(self, 'freq_4d', False):
                x = freq(x)  # H3 modules work on the (b, t, band, dim) grid: frame pooling needs t
            else:
                _, _, f, d = x.shape
                x = freq(x.reshape(b * t, f, d)).reshape(b, t, f, d)
        return mask_to_complex_shape(self._estimate_masks(self.final_norm(x)))


def factory(*args, **kw):
    arm = resolve(os.environ['ARM']); assert arm == 'A' or arm in ARMS, arm
    if arm == 'H2N128':
        from narrow_h2 import build_narrow
        config_path = args[1] if len(args) > 1 else kw.get('config_path', CFG)
        model, cfg = build_narrow(128, 2, config_path)
        model.arm = arm
        return model, cfg
    if arm == 'BandUNet':
        from band_unet import build as build_band_unet
        config_path = args[1] if len(args) > 1 else kw.get('config_path', CFG)
        return build_band_unet(str(config_path))
    m, cfg = get_model_from_config(*args, **kw)
    assert len(m.layers) == 6 and len(m.band_split.to_features) == 62 and not m.skip_connection
    if arm != 'A':
        spec = ARMS[arm]
        m.__class__ = ArmModel; m.arm = arm
        for layer in m.layers:
            layer[0] = nn.Identity()
        with torch.random.fork_rng(devices=[]):
            cv = MSC(dim=192, mult=spec['mult'], dilations=spec['dilations'], k=31, dropout=cfg.model.ff_dropout, gate_dropout=spec['gate_dropout'])
        m.convs = nn.ModuleList(copy.deepcopy(cv) for _ in range(6))
        m.temporal_norms = nn.ModuleList(RMSNorm(192) for _ in range(6))
        if arm.startswith('H2') or arm == 'H3':
            rotary = m.layers[0][1].layers[0][0].rotary_embed
            # Share the fixed RoPE table with frequency attention and preserve the
            # training RNG state and initialization of the four convolution blocks.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(6204)
                for i in H2_ATTENTION_LAYERS:
                    m.convs[i] = PooledKVAttention(192, rotary, cfg.model.attn_dropout)
        if arm in ('H2M', 'H2MG'):
            with torch.random.fork_rng(devices=[]):
                for i, (_, frequency) in enumerate(m.layers):
                    block = frequency.layers[0]
                    torch.manual_seed(450 + i)
                    block[0] = BandMixer(block[0].norm)
                    if arm == 'H2MG':
                        torch.manual_seed(950 + i)
                        block[1] = BandFFN(block[1])
        if arm == 'H3':
            from h3 import apply_h3
            apply_h3(m, cfg)
    return m, cfg


def build(arm):
    arm = resolve(arm)
    os.environ['ARM'] = arm; torch.manual_seed(0); return factory('bs_roformer', CFG)[0]


def structure_ok(m):
    spec = ARMS[m.arm]
    if m.arm == 'BandUNet':
        return (len(m.layers) == 0 and len(m.temporal) == len(m.bottleneck) == 4
                and tuple(module.bands_out for module in (m.down32, m.down16, m.down8)) == (32, 16, 8)
                and tuple(module.bands_out for module in (m.up16, m.up32, m.up62)) == (16, 32, 62)
                and sum(p.numel() for p in m.parameters()) == spec['params'])
    if m.arm == 'H2N128':
        return (m.stride == 2 and m.downsample.in_features == m.upsample.out_features == 256
                and len(m.layers) == len(m.convs) == len(m.temporal_norms) == 6
                and len(m.band_split.to_features) == 62
                and tuple(i for i, block in enumerate(m.convs) if isinstance(block, PooledKVAttention)) == H2_ATTENTION_LAYERS
                and sum(p.numel() for p in m.parameters()) == spec['params'])
    ok = len(m.convs) == 6 and len(m.temporal_norms) == 6 and all(isinstance(layer[0], nn.Identity) for layer in m.layers)
    for i, cv in enumerate(m.convs):
        if (m.arm.startswith('H2') or m.arm == 'H3') and i in H2_ATTENTION_LAYERS:
            ok &= isinstance(cv, PooledKVAttention) and (cv.heads, cv.head_dim, cv.stride) == (2, 64, 4)
            continue
        if not isinstance(cv, MSC):
            return False
        h = 192 * spec['mult'] // 2
        ok &= len(cv.convs) == len(spec['dilations']) and [int(c.dilation[0]) for c in cv.convs] == list(spec['dilations'])
        ok &= all(tuple(c.kernel_size) == (31,) and c.groups == h and c.in_channels == h and c.padding[0] == 15 * c.dilation[0] for c in cv.convs)
        ok &= cv.proj_in.in_features == 192 and cv.proj_in.out_features == 2 * h and cv.proj_out.in_features == h and cv.proj_out.out_features == 192
        ok &= isinstance(cv.gate_drop, nn.Dropout) == spec['gate_dropout'] and cv.swiglu
    for i, (_, frequency) in enumerate(m.layers):
        mixer, feedforward = frequency.layers[0]
        ok &= isinstance(mixer, BandMixer) == (m.arm in ('H2M', 'H2MG'))
        ok &= isinstance(feedforward, BandFFN) == (m.arm == 'H2MG')
        if m.arm == 'H3':
            from h3 import BlockFeedForward, STATIC_STACKS, STRIDE, V_WIDTH, SlowBandAttention, StaticBandMixer
            if i < STATIC_STACKS:
                ok &= isinstance(mixer, StaticBandMixer) and type(feedforward) is FeedForward
            else:
                ok &= (isinstance(mixer, SlowBandAttention) and mixer.rotary_embed is not None
                       and (mixer.stride, mixer.to_v.out_features, mixer.to_qk.out_features, mixer.v_dim) == (STRIDE, V_WIDTH, 1024, V_WIDTH // 8)
                       and isinstance(feedforward, BlockFeedForward) and feedforward.stride == STRIDE)
    ok &= sum(p.numel() for p in m.parameters()) == spec['params']
    return ok


def autocast():
    """bf16 autocast on the training device (CUDA on Colab, Ascend NPU via torch_npu); disabled elsewhere."""
    return torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type in ('cuda', 'npu'))


def empty_cache():
    if DEVICE.type == 'cuda':
        empty_cache()
    elif DEVICE.type == 'npu':
        torch.npu.empty_cache()


def install_npu_trainer():
    """Ascend: Lightning 2.6 ships no NPU accelerator.  Register one (CUDAAccelerator's contract) and make the Trainer
    select it: one NPU, bf16 autocast typed for the npu device.  Never runs when DEVICE is not an NPU."""
    import pytorch_lightning as pl
    from pytorch_lightning.accelerators import Accelerator, AcceleratorRegistry
    from pytorch_lightning.plugins import MixedPrecision   # PL 2.6 wants its own subclass, not lightning_fabric's
    from pytorch_lightning.strategies import SingleDeviceStrategy

    class NPUAccelerator(Accelerator):
        accelerator_name = 'npu'

        def setup_device(self, device):
            assert device.type == 'npu', device
            torch.npu.set_device(device)

        def setup(self, trainer): pass
        def teardown(self): torch.npu.empty_cache()
        def get_device_stats(self, device): return {}
        @staticmethod
        def parse_devices(devices): return list(range(devices)) if isinstance(devices, int) else [int(d) for d in devices]
        @staticmethod
        def get_parallel_devices(devices): return [torch.device('npu', i) for i in devices]
        @staticmethod
        def auto_device_count(): return torch.npu.device_count()
        @staticmethod
        def is_available(): return torch.npu.is_available()
        @staticmethod
        def name(): return 'npu'

    AcceleratorRegistry.register('npu', NPUAccelerator, description='Ascend NPU (torch_npu)')
    original = pl.Trainer

    class Trainer(original):
        def __init__(self, *args, **kwargs):
            if kwargs.get('accelerator') in ('gpu', 'npu'):   # stage.py asks for the CUDA flags; run them on the NPU
                kwargs.update(accelerator='npu', devices=1, strategy=SingleDeviceStrategy(device=DEVICE))
                if kwargs.get('precision') == 'bf16-mixed':
                    del kwargs['precision']
                    assert not kwargs.get('plugins'), kwargs.get('plugins')
                    kwargs['plugins'] = [MixedPrecision('bf16-mixed', 'npu')]
            super().__init__(*args, **kwargs)

    pl.Trainer = Trainer


def check():
    x = torch.randn(1, 2, 131584, device=DEVICE)
    for arm in ('A',) + tuple(ARMS):
        m = build(arm).to(DEVICE)
        if arm == 'BandUNet':
            with autocast():
                y = m(x)
            assert y.shape == x.shape and torch.isfinite(y).all()
            y.float().square().mean().backward()
            n = sum(p.numel() for p in m.parameters())
            assert n == ARMS[arm]['params'] and structure_ok(m), (arm, n)
            for mod in (m.enc62, m.enc32, m.enc16, m.temporal, m.bottleneck, m.dec16, m.dec32, m.dec62):
                for name, p in mod.named_parameters():
                    if p.requires_grad:
                        assert p.grad is not None and torch.isfinite(p.grad).all(), (arm, name)
            print('arm', arm, 'params', n, 'max|y|', float(y.abs().max()), flush=True)
            del m; empty_cache()
            continue
        order = []; handles = []
        if arm != 'A':
            handles += [layer[1].register_forward_pre_hook(lambda *_: order.append('F')) for layer in m.layers]
            handles += [cv.register_forward_pre_hook(lambda *_: order.append('T')) for cv in m.convs]
        with autocast():
            y = m(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.float().square().mean().backward()
        n = sum(p.numel() for p in m.parameters())
        assert n == (A_PARAMS if arm == 'A' else ARMS[arm]['params']), (arm, n)
        if arm != 'A':
            assert ''.join(order) == 'TF' * 6, order   # event order per stack: temporal conv, then frequency transformer
            assert structure_ok(m), arm
            checked = list(m.convs) + list(m.temporal_norms)
            if arm in ('H2M', 'H2MG', 'H2N128', 'H3'):
                checked += [frequency for _, frequency in m.layers]
            if arm == 'H2N128':
                checked += [m.downsample, m.upsample]
            for mod in checked:
                for name, p in mod.named_parameters():
                    if p.requires_grad:
                        assert p.grad is not None and torch.isfinite(p.grad).all(), (arm, name)
        for h in handles:
            h.remove()
        print('arm', arm, 'params', n, 'max|y|', float(y.abs().max()), flush=True)
        del m; empty_cache()
    # structural equivalence: S2 layout restricted to a single d=1 branch reproduces S exactly
    torch.manual_seed(1); z = torch.randn(2, 64, 62, 192)
    s = build('S').eval(); single = MSC(192, 4, (1,)).eval()
    with torch.no_grad():
        single.load_state_dict({k.replace('.conv.', '.convs.0.'): v for k, v in s.convs[0].state_dict().items()}, strict=True)
        torch.testing.assert_close(single(z), s.convs[0](z), rtol=0, atol=0)
    print('PREFLIGHT_PASS S exact under single-branch layout', flush=True)


def install_resume_hook():
    from pymsst.training.module import SeparationLightningModule as LM
    from pymsst.training.callbacks import ExponentialMovingAverage as EMA
    old_load, old_fit, old_start = LM.on_load_checkpoint, EMA.on_fit_start, EMA.on_train_start
    def load(module, ck):
        old_load(module, ck)
        sd = ck['state_dict']; keys = [k for k in sd if k.startswith('ema_model.')]
        if ck['global_step']:
            assert keys and ck['optimizer_states']
            if module.ema_model is None:
                module._resume_ema_sd = {k[len('ema_model.'):]: sd.pop(k) for k in keys}
            module._resume_expected = (int(ck['global_step']), int(ck['state_dict']['ema_model.n_averaged'] if module.ema_model is not None else module._resume_ema_sd['n_averaged']))
    def fit(cb, trainer, module):
        old_fit(cb, trainer, module)
        if hasattr(module, '_resume_ema_sd'):
            cb.ema_model.load_state_dict(module._resume_ema_sd, strict=True); del module._resume_ema_sd
    def start(cb, trainer, module):
        old_start(cb, trainer, module)
        if hasattr(module, '_resume_expected'):
            step, avg = module._resume_expected; adam_steps = {int(s['step']) for s in trainer.optimizers[0].state.values() if 'step' in s}
            assert trainer.global_step == step and int(cb.ema_model.n_averaged) == avg and max(adam_steps) == step, (trainer.global_step, step, avg, adam_steps)
            print('RESUME_VERIFIED', step, 'EMA', avg, 'Adam', sorted(adam_steps), flush=True)
    LM.on_load_checkpoint, EMA.on_fit_start, EMA.on_train_start = load, fit, start


if __name__ == '__main__':
    if sys.argv[1:] == ['check']:
        check()
    else:
        raise SystemExit('use s2_aw_entry.py for training')
