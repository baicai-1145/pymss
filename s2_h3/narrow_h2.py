"""Reduced-rate H2 backbone with learned time-frame packing and a full-rate skip."""
import copy

from torch import nn
from torch.nn import functional as F

from entry import ArmModel, CFG, MSC, get_model_from_config, mask_to_complex_shape
from pooled_kv import PooledKVAttention
from pymss_core.modules.bs_roformer.common import RotaryEmbedding
from pymss_core.modules.bs_roformer.transformer import RMSNorm


class NarrowH2(ArmModel):
    def _forward_mask_core(self, stft):
        if self.stride == 1:
            return super()._forward_mask_core(stft)
        batch, freq, frames, components = stft.shape
        features = stft.transpose(1, 2).reshape(batch, frames, freq * components)
        bands = self.band_split(features.to(next(self.band_split.parameters()).dtype))
        pad = (-frames) % self.stride
        padded = F.pad(bands, (0, 0, 0, 0, 0, pad)) if pad else bands
        group = padded.unflatten(1, (-1, self.stride)).transpose(2, 3).flatten(3)
        hidden = self.downsample(group)
        _, reduced_frames, n_bands, dim = hidden.shape
        for conv, norm, (_, frequency) in zip(self.convs, self.temporal_norms, self.layers):
            hidden = hidden + conv(norm(hidden))
            hidden = frequency(hidden.reshape(-1, n_bands, dim)).reshape(batch, reduced_frames, n_bands, dim)
        detail = self.upsample(hidden).unflatten(-1, (self.stride, dim)).transpose(2, 3).flatten(1, 2)
        return mask_to_complex_shape(self._estimate_masks(self.final_norm(bands + detail[:, :frames])))


def build_narrow(width, stride, config_path=CFG):
    assert width % 32 == 0 and stride >= 1
    overrides = {'dim': width, 'heads': width // 32, 'dim_head': 32, 'ff_mult': 2}
    model, cfg = get_model_from_config('bs_roformer', config_path, model_kwargs_override=overrides)
    cfg.model.update(overrides)
    model.__class__ = NarrowH2
    model.stride = stride
    if stride > 1:
        model.downsample = nn.Linear(width * stride, width)
        model.upsample = nn.Linear(width, width * stride)
    conv = MSC(width, mult=2, k=31, dropout=cfg.model.ff_dropout)
    rotary = RotaryEmbedding(64)
    model.convs = nn.ModuleList(PooledKVAttention(width, rotary, cfg.model.attn_dropout)
                                if index in (0, 4) else copy.deepcopy(conv) for index in range(6))
    model.temporal_norms = nn.ModuleList(RMSNorm(width) for _ in model.layers)
    for layer in model.layers:
        layer[0] = nn.Identity()
    return model, cfg
