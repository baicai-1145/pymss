"""Target-only frequency U-Net prototype for FP32 shape, gradient and latency checks."""
from copy import deepcopy
from pathlib import Path
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from entry import MSC  # noqa: E402
from pymss_core import get_model_from_config  # noqa: E402
from pymss_core.modules.bs_roformer import transformer as tr  # noqa: E402
from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer  # noqa: E402
from pymss_core.modules.bs_roformer.common import RotaryEmbedding, mask_to_complex_shape  # noqa: E402
from pymss_core.modules.bs_roformer.transformer import RMSNorm  # noqa: E402


class LocalBandBlock(nn.Module):
    """Cheap shifted-free local frequency mixing at one resolution."""

    def __init__(self, dim, kernel=7, mult=2):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.band = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.mix = nn.Linear(dim, dim)
        self.ff_norm = RMSNorm(dim)
        self.ff_in = nn.Linear(dim, dim * mult * 2)
        self.ff_out = nn.Linear(dim * mult, dim)

    def forward(self, x):
        batch, frames, bands, width = x.shape
        y = self.norm(x).permute(0, 1, 3, 2).reshape(batch * frames, width, bands)
        y = self.band(y).transpose(1, 2).reshape(batch, frames, bands, width)
        x = x + self.mix(y)
        y = self.ff_norm(x)
        u, v = self.ff_in(y).chunk(2, dim=-1)
        return x + self.ff_out(u * F.silu(v))


class BandDown(nn.Module):
    def __init__(self, dim, bands_in, bands_out):
        super().__init__()
        self.bands_in, self.bands_out = bands_in, bands_out
        self.proj = nn.Conv1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        batch, frames, bands, width = x.shape
        pad = 1 if bands == 62 else 0
        y = F.pad(x.permute(0, 1, 3, 2).reshape(batch * frames, width, bands), (pad, pad))
        y = self.proj(y).transpose(1, 2)
        assert y.shape[1] == self.bands_out, y.shape
        return y.reshape(batch, frames, self.bands_out, width)


class BandUp(nn.Module):
    def __init__(self, dim, bands_out):
        super().__init__()
        self.bands_out = bands_out
        self.proj = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x, skip):
        batch, frames, bands, width = x.shape
        y = self.proj(x.permute(0, 1, 3, 2).reshape(batch * frames, width, bands))
        y = y[..., :self.bands_out].transpose(1, 2).reshape(batch, frames, self.bands_out, width)
        assert y.shape == skip.shape, (y.shape, skip.shape)
        return y + skip * self.gate(skip).sigmoid()


class BandUNet(BSRoformer):
    def _frequency(self, x):
        batch, frames, _, width = x.shape
        skips = []
        x = self.enc62(x)
        skips.append(x)
        x = self.down32(x)
        x = self.enc32(x)
        skips.append(x)
        x = self.down16(x)
        x = self.enc16(x)
        skips.append(x)
        x = self.down8(x)
        for temporal, frequency in zip(self.temporal, self.bottleneck):
            x = x + temporal(x)
            y = frequency(x.reshape(batch * frames, 8, width))
            x = x + y.reshape(batch, frames, 8, width)
        x = self.up16(x, skips[2])
        x = self.dec16(x)
        x = self.up32(x, skips[1])
        x = self.dec32(x)
        x = self.up62(x, skips[0])
        return self.dec62(x)

    def _forward_mask_core(self, stft):
        batch, frequencies, frames, complex_dim = stft.shape
        features = stft.permute(0, 2, 1, 3).reshape(batch, frames, frequencies * complex_dim)
        dtype = next(self.band_split.parameters()).dtype
        x = self.band_split(features.to(dtype=dtype))
        x = self._frequency(x)
        return mask_to_complex_shape(self._estimate_masks(self.final_norm(x)))


def build(config_path=None):
    config_path = config_path or str(ROOT / 'aw.yaml')
    model, config = get_model_from_config('bs_roformer', config_path)
    assert isinstance(model, BSRoformer)
    model.__class__ = BandUNet
    model.layers = nn.ModuleList()
    dim = 192
    model.enc62 = LocalBandBlock(dim)
    model.down32 = BandDown(dim, 62, 32)
    model.enc32 = LocalBandBlock(dim)
    model.down16 = BandDown(dim, 32, 16)
    model.enc16 = LocalBandBlock(dim)
    model.down8 = BandDown(dim, 16, 8)
    model.temporal = nn.ModuleList(MSC(dim=dim, mult=2, dilations=(1, 4, 16), k=31, dropout=config.model.ff_dropout) for _ in range(4))
    rotary = RotaryEmbedding(64)
    model.bottleneck = nn.ModuleList(tr.Transformer(dim=dim, depth=1, heads=2, dim_head=64,
                                                     attn_dropout=config.model.attn_dropout,
                                                     ff_dropout=config.model.ff_dropout,
                                                     rotary_embed=rotary, flash_attn=True) for _ in range(4))
    model.up16 = BandUp(dim, 16)
    model.dec16 = LocalBandBlock(dim)
    model.up32 = BandUp(dim, 32)
    model.dec32 = LocalBandBlock(dim)
    model.up62 = BandUp(dim, 62)
    model.dec62 = LocalBandBlock(dim)
    model.arm = 'BandUNet'
    return model, config


def check():
    torch.manual_seed(19)
    model, _ = build()
    model.train()
    wave = torch.randn(1, 2, 352800)
    output = model(wave)
    assert output.shape == wave.shape and output.isfinite().all(), output.shape
    loss = output.square().mean()
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None and p.grad.isfinite().all() for p in trainable)
    print('params', sum(p.numel() for p in model.parameters()))
    print('output', tuple(output.shape), 'loss', float(loss), 'gradients', len(trainable))


if __name__ == '__main__':
    if os.environ.get('BAND_UNET_CHECK'):
        check()
