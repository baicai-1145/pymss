"""band U-Net v2: the three mechanisms that make a resolution pyramid keep information.

v1 borrowed only the shape of the dlssnr-pytorch pyramid. v2 adds what it left out:
  1. process before pooling: every downsample runs a block at the fine resolution first, and the
     channel count doubles as the band count halves (192 -> 384 -> 768 -> 1536)
  2. the band grid shifts between blocks, so no group boundary is permanent
  3. pooling is a plain average; the learned part is the projection after it
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

LEVELS = (62, 32, 16, 8)
WIDTHS = (192, 384, 768, 1536)
WINDOW = 8


class BandBlock(nn.Module):
    """One block at one band resolution: windowed band attention with a shifted grid, then SwiGLU.

    ``shift`` moves the band grid by half a window, so the partitions of consecutive blocks differ
    and information crosses the window edges a block could not see past.
    """

    def __init__(self, dim, bands, heads, shift=0, mult=2, dropout=0.0, window=WINDOW):
        super().__init__()
        self.bands, self.window, self.shift = bands, min(window, bands), shift
        self.heads, self.head_dim = heads, dim // heads
        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.pos = nn.Parameter(torch.zeros(heads, self.window * 2))
        self.drop = nn.Dropout(dropout)
        self.ff_norm = RMSNorm(dim)
        self.ff_in = nn.Linear(dim, dim * mult * 2)
        self.ff_out = nn.Linear(dim * mult, dim)

    def _windows(self, x):
        """Split the band axis into windows of ``self.window`` bands, on a grid shifted by half a window when
        ``self.shift`` is set (the bands that fall off one end wrap to the other). Returns the windows
        ``[batch * frames * groups, window, width]`` and the original band count."""
        batch, frames, bands, width = x.shape
        w = self.window
        half = w // 2 if self.shift and bands > w else 0
        if half:
            x = torch.cat([x[..., bands - half:, :], x, x[..., :half, :]], dim=2)
        pad = (-x.shape[2]) % w
        if pad:
            x = F.pad(x, (0, 0, 0, pad))  # (W_left, W_right, band_front, band_back)
        groups = x.shape[2] // w
        return x.reshape(batch * frames, groups, w, width).reshape(batch * frames * groups, w, width), bands, half, pad

    @staticmethod
    def _unwindows(y, batch, frames, bands, half, pad, width):
        """Undo ``_windows``: fold windows back to the band axis and unwrap the shifted bands."""
        y = y.reshape(batch * frames, -1, width)[:, half:half + bands]
        return y.reshape(batch, frames, bands, width)

    def forward(self, x):
        batch, frames, bands, width = x.shape
        tokens, bands, half, pad = self._windows(self.norm(x))
        groups = tokens.shape[0]
        qkv = self.qkv(tokens).reshape(groups, self.window, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        rel = torch.arange(self.window)
        index = (rel[:, None] - rel[None, :]) % (self.window * 2)
        logits = torch.einsum('ghqd,ghkd->ghqk', q, k) / self.head_dim ** 0.5 + self.pos[:, index]
        attended = torch.softmax(logits, dim=-1) @ v
        attended = attended.transpose(1, 2).reshape(groups, self.window, -1)
        attended = self.out(attended)
        attended = self._unwindows(attended, batch, frames, bands, half, pad, width)
        x = x + self.drop(attended)
        y = self.ff_norm(x)
        u, v = self.ff_in(y).chunk(2, dim=-1)
        return x + self.drop(self.ff_out(u * F.silu(v)))


class BandDown(nn.Module):
    """Process at the fine resolution, average pool pairs of bands, then project to twice the width."""

    def __init__(self, dim_in, dim_out, bands, heads, window=WINDOW):
        super().__init__()
        self.block = BandBlock(dim_in, bands, heads, shift=1, window=window)
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.block(x)
        bands = x.shape[2]
        # ceil division like dlssnr's pooled extent: an odd band count pads its last group by zeros
        pad = bands % 2
        if pad:
            x = F.pad(x, (0, 0, 0, 1))
        # x is (batch, width, frames, bands) here: pool pairs of bands, leave frames alone
        return self.proj(F.avg_pool2d(x.permute(0, 3, 1, 2), (1, 2)).permute(0, 2, 3, 1))


class BandUp(nn.Module):
    """Project to the skip's width, repeat each band twice, gate the skip from the fine level."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.gate = nn.Linear(dim_out, dim_out)

    def forward(self, x, skip):
        x = self.proj(x).repeat_interleave(2, dim=2)[:, :, :skip.shape[2]]
        return x + skip * torch.sigmoid(self.gate(skip))


class BandUNet2(BSRoformer):
    def _frequency(self, x):
        batch, frames = x.shape[:2]
        skips = []
        for down in self.downs:
            skips.append(x)  # the input of each level, at that level's width: 192, 384, 768
            x = down(x)
        width = x.shape[-1]
        for temporal, frequency in zip(self.temporal, self.bottleneck):
            x = x + temporal(x)
            y = frequency(x.reshape(batch * frames, 8, width))
            x = x + y.reshape(batch, frames, 8, width)
        for up, block in zip(self.ups, self.decoders):
            x = block(up(x, skips.pop()))  # pops 768, 384, 192: the level just below each upsample
        return x

    def _forward_mask_core(self, stft):
        batch, frequencies, frames, complex_dim = stft.shape
        features = stft.permute(0, 2, 1, 3).reshape(batch, frames, frequencies * complex_dim)
        dtype = next(self.band_split.parameters()).dtype
        x = self.band_split(features.to(dtype=dtype))
        x = self.band_split_proj(x)
        x = self._frequency(x)
        return mask_to_complex_shape(self._estimate_masks(self.final_norm(self.out_proj(x))))


def build(config_path=None, widths=WIDTHS, heads=(2, 3, 6, 12), bottleneck=4, dropout=None):
    config_path = config_path or str(ROOT / 'aw.yaml')
    model, config = get_model_from_config('bs_roformer', config_path)
    assert isinstance(model, BSRoformer)
    model.__class__ = BandUNet2
    model.layers = nn.ModuleList()
    dropout = config.model.ff_dropout if dropout is None else dropout
    model.band_split_proj = nn.Linear(192, widths[0])
    model.downs = nn.ModuleList()
    for level in range(3):
        model.downs.append(BandDown(widths[level], widths[level + 1], LEVELS[level], heads[level]))
    model.temporal = nn.ModuleList(MSC(dim=widths[-1], mult=2, dilations=(1, 4, 16), k=31, dropout=dropout) for _ in range(bottleneck))
    rotary = RotaryEmbedding(64)
    model.bottleneck = nn.ModuleList(tr.Transformer(dim=widths[-1], depth=1, heads=heads[-1], dim_head=64,
                                                     attn_dropout=config.model.attn_dropout,
                                                     ff_dropout=dropout, rotary_embed=rotary, flash_attn=True) for _ in range(bottleneck))
    model.ups = nn.ModuleList()
    model.decoders = nn.ModuleList()
    for level in (2, 1, 0):
        model.ups.append(BandUp(widths[level + 1], widths[level]))
        model.decoders.append(BandBlock(widths[level], LEVELS[level], heads[level], shift=0, dropout=dropout))
    model.out_proj = nn.Linear(widths[0], 192)
    model.arm = 'BandUNet2'
    return model, config


def count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for n, p in model.named_parameters() if n.startswith(('band_split_proj', 'downs')))
    middle = sum(p.numel() for n, p in model.named_parameters() if n.startswith(('temporal', 'bottleneck')))
    decoder = sum(p.numel() for n, p in model.named_parameters() if n.startswith(('ups', 'decoders', 'out_proj')))
    return dict(total=total, trainable=trainable, encoder=encoder, bottleneck=middle, decoder=decoder,
                inherited=total - encoder - middle - decoder)


def check(model=None):
    torch.manual_seed(19)
    model, _ = build() if model is None else (model, None)
    model.train()
    wave = torch.randn(1, 2, 352800)
    output = model(wave)
    assert output.shape == wave.shape and output.isfinite().all(), output.shape
    output.square().mean().backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and p.grad.isfinite().all() for p in trainable)
    return dict(params=count(model), output=tuple(output.shape), gradients=len(trainable))


if __name__ == '__main__':
    import json
    print(json.dumps(check()))
