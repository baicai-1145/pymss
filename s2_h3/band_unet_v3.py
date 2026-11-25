"""band U-Net v3: v1's parameter budget, spent where v1 spent none.

v1 had 21M parameters and no cross-band communication at the 62/32/16-band levels: every fine-resolution layer was a
depthwise conv over +-3 bands, and the only attention lived on the 8-band bottleneck. H2 spends its 21M on six
layers of full attention over all 62 bands. v3 moves the budget the same way: windowed band attention with a
shifted grid at every encoder and decoder level, paid for by halving the bottleneck and its temporal convolutions.
Width stays 192 throughout (the time axis is not downsampled, so widening costs dim^2), and downsampling is
process-then-average-pool with a projection, as in dlssnr-pytorch.
"""
from pathlib import Path
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
from band_unet_v2 import BandBlock, BandDown, BandUp  # noqa: E402

LEVELS = (62, 31, 16, 8)
WIDTH = 192


class BandUNet3(BSRoformer):
    def _frequency(self, x):
        batch, frames = x.shape[:2]
        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x)
        for temporal, frequency in zip(self.temporal, self.bottleneck):
            x = x + temporal(x)
            y = frequency(x.reshape(batch * frames, 8, WIDTH))
            x = x + y.reshape(batch, frames, 8, WIDTH)
        for up, block in zip(self.ups, self.decoders):
            x = block(up(x, skips.pop()))
        return x

    def _forward_mask_core(self, stft):
        batch, frequencies, frames, complex_dim = stft.shape
        features = stft.permute(0, 2, 1, 3).reshape(batch, frames, frequencies * complex_dim)
        dtype = next(self.band_split.parameters()).dtype
        x = self.band_split(features.to(dtype=dtype))
        return mask_to_complex_shape(self._estimate_masks(self.final_norm(self._frequency(x))))


def build(config_path=None, heads=(2, 2, 3, 3), bottleneck=2, msc=1, window=8, dim=WIDTH, dropout=None):
    config_path = config_path or str(ROOT / 'aw.yaml')
    model, config = get_model_from_config('bs_roformer', config_path)
    assert isinstance(model, BSRoformer)
    model.__class__ = BandUNet3
    model.layers = nn.ModuleList()
    dropout = config.model.ff_dropout if dropout is None else dropout
    model.downs = nn.ModuleList()
    for level in range(3):
        model.downs.append(BandDown(dim, dim, LEVELS[level], heads[level], window=window))
    model.temporal = nn.ModuleList(MSC(dim=dim, mult=2, dilations=(1, 4, 16), k=31, dropout=dropout) for _ in range(msc))
    rotary = RotaryEmbedding(64)
    model.bottleneck = nn.ModuleList(tr.Transformer(dim=dim, depth=1, heads=heads[3], dim_head=64,
                                                     attn_dropout=config.model.attn_dropout,
                                                     ff_dropout=dropout, rotary_embed=rotary, flash_attn=True) for _ in range(bottleneck))
    model.ups = nn.ModuleList()
    model.decoders = nn.ModuleList()
    for level in (2, 1, 0):
        model.ups.append(BandUp(dim, dim))
        model.decoders.append(BandBlock(dim, LEVELS[level], heads[level], shift=0, dropout=dropout, window=window))
    model.arm = 'BandUNet3'
    return model, config


def count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = dict(encoder='downs', bottleneck='bottleneck', temporal='temporal', decoder='decoders', ups='ups')
    parts = {name: sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix)) for name, prefix in groups.items()}
    parts['inherited'] = total - sum(parts.values())
    return dict(total=total, trainable=trainable, **parts)


if __name__ == '__main__':
    import json
    torch.manual_seed(19)
    model, _ = build(dropout=0.0)
    model.train()
    wave = torch.randn(1, 2, 352800)
    output = model(wave)
    assert output.shape == wave.shape and output.isfinite().all(), output.shape
    output.square().mean().backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and p.grad.isfinite().all() for p in trainable)
    print(json.dumps(dict(params=count(model), output=tuple(output.shape), gradients=len(trainable))))
