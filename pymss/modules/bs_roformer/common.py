from functools import partial
from typing import NamedTuple

import torch
from torch import nn
from rotary_embedding_torch import RotaryEmbedding

from .bands import BandSplit, MaskEstimator
from .transformer import RMSNorm, Transformer


__all__ = [
    'DEFAULT_FREQS_PER_BANDS',
    'MaskEstimator',
    'RMSNorm',
    'RoformerRuntimeMixin',
    'forward_bandsplit_roformer',
    'forward_roformer_mask_core',
    'forward_spectral_roformer',
    'ignore_roformer_training_kwargs',
    'init_roformer_band_modules',
    'init_roformer_layers',
    'init_roformer_runtime',
    'init_roformer_stft',
    'roformer_freqs_per_bands_with_complex',
]


DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


class SpectralContext(NamedTuple):
    batch: int
    channels: int
    freq_bins: int
    audio_length: int
    stft_window: torch.Tensor
    x_is_mps: bool


def default(v, d):
    return v if v is not None else d


def mask_to_complex_shape(mask, complex_dim=2):
    b, n, t, fc = mask.shape
    return mask.reshape(b, n, t, fc // complex_dim, complex_dim).permute(0, 1, 3, 2, 4)


TRAINING_LOSS_KWARGS = frozenset({
    'multi_stft_resolution_loss_weight',
    'multi_stft_resolutions_window_sizes',
    'multi_stft_hop_size',
    'multi_stft_normalized',
    'multi_stft_window_fn',
})

REMOVED_ROFORMER_KWARGS = frozenset({
    'linear_transformer_depth',
    'use_torch_checkpoint',
    'skip_connection',
    'sage_attention',
    'sage_attention_mode',
    'attention_layout',
    'dim_freqs_in',
})


def ignore_roformer_training_kwargs(kwargs):
    unexpected = set(kwargs) - TRAINING_LOSS_KWARGS - REMOVED_ROFORMER_KWARGS
    if unexpected:
        raise TypeError(f"unexpected RoFormer config keys: {sorted(unexpected)}")


def init_roformer_runtime(module, stereo, num_stems):
    module.stereo = stereo
    module.audio_channels = 2 if stereo else 1
    module.num_stems = num_stems


def init_roformer_layers(
        module,
        *,
        depth,
        time_transformer_depth,
        freq_transformer_depth,
        dim_head,
        transformer_kwargs,
):
    module.layers = nn.ModuleList([])
    time_rotary_embed = RotaryEmbedding(dim=dim_head)
    freq_rotary_embed = RotaryEmbedding(dim=dim_head)

    for _ in range(depth):
        module.layers.append(nn.ModuleList([
            Transformer(
                depth=time_transformer_depth,
                rotary_embed=time_rotary_embed,
                **transformer_kwargs
            ),
            Transformer(
                depth=freq_transformer_depth,
                rotary_embed=freq_rotary_embed,
                **transformer_kwargs
            ),
        ]))


def init_roformer_stft(module, stft_n_fft, stft_hop_length, stft_win_length, stft_normalized, stft_window_fn):
    module.stft_kwargs = dict(
        n_fft=stft_n_fft,
        hop_length=stft_hop_length,
        win_length=stft_win_length,
        normalized=stft_normalized,
    )
    module.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)
    module._stft_window_cache = {}


def roformer_freqs_per_bands_with_complex(module, freqs_per_bands, freqs):
    assert len(freqs_per_bands) > 1
    assert sum(
        freqs_per_bands
    ) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'
    return tuple(2 * f * module.audio_channels for f in freqs_per_bands)


def init_roformer_band_modules(
        module,
        *,
        dim,
        freqs_per_bands_with_complex,
        num_stems,
        mask_estimator_cls,
        mask_estimator_depth,
        mlp_expansion_factor,
        mask_estimator_kwargs=None,
):
    module.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)
    module.mask_estimators = nn.ModuleList([
        mask_estimator_cls(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex,
            depth=mask_estimator_depth,
            mlp_expansion_factor=mlp_expansion_factor,
            **(mask_estimator_kwargs or {}),
        )
        for _ in range(num_stems)
    ])


class RoformerRuntimeMixin:
    def stft_window(self, device):
        key = (device.type, device.index, torch.float32)
        window = self._stft_window_cache.get(key)
        if window is None or window.device != device:
            window = self.stft_window_fn(device=device)
            self._stft_window_cache[key] = window
        return window

    def _warm_group_cache(self, tensor):
        self.band_split.warm_group_cache(tensor.device, tensor.dtype)
        for mask_estimator in self.mask_estimators:
            mask_estimator.warm_group_cache(tensor.device, tensor.dtype)

    def _estimate_masks(self, x):
        return torch.stack([fn(x) for fn in self.mask_estimators], dim=1)

    def _mask_stft_repr(self, stft_repr, context):
        self._warm_group_cache(stft_repr)
        mask = self._forward_mask_core(stft_repr)
        stft_repr = torch.view_as_complex(stft_repr.unsqueeze(1))
        mask = torch.view_as_complex(mask.contiguous()).type(stft_repr.dtype)
        return stft_repr * mask


def forward_roformer_mask_core(module, stft_repr):
    b, fs, model_t, complex_dim = stft_repr.shape
    x = stft_repr.permute(0, 2, 1, 3).reshape(b, model_t, fs * complex_dim)
    x = module.band_split(x)

    for time_transformer, freq_transformer in module.layers:
        b, t, f, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
        x = time_transformer(x)
        x = x.reshape(b, f, t, d).permute(0, 2, 1, 3)

        x = x.reshape(b * t, f, d)
        x = freq_transformer(x)
        x = x.reshape(b, t, f, d)

    x = module.final_norm(x)
    mask = module._estimate_masks(x)
    return mask_to_complex_shape(mask, complex_dim=2)


def stft_roformer(module, raw_audio):
    device = raw_audio.device
    x_is_mps = device.type == "mps"

    if raw_audio.ndim == 2:
        raw_audio = raw_audio.unsqueeze(1)

    batch, audio_channels, audio_length = raw_audio.shape
    assert (
        not module.stereo and audio_channels == 1
    ) or (
        module.stereo and audio_channels == 2
    ), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

    stft_audio = raw_audio.reshape(batch * audio_channels, audio_length)
    stft_window = module.stft_window(device)

    try:
        stft_repr = torch.stft(stft_audio, **module.stft_kwargs, window=stft_window, return_complex=True)
    except RuntimeError:
        stft_repr = torch.stft(
            stft_audio.cpu() if x_is_mps else stft_audio,
            **module.stft_kwargs,
            window=stft_window.cpu() if x_is_mps else stft_window,
            return_complex=True
        ).to(device)

    stft_repr = torch.view_as_real(stft_repr)
    stft_repr = stft_repr.reshape(batch, audio_channels, *stft_repr.shape[-3:])

    b, s, f, t, c = stft_repr.shape
    stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(b, f * s, t, c)
    context = SpectralContext(
        batch=batch,
        channels=audio_channels,
        freq_bins=f,
        audio_length=audio_length,
        stft_window=stft_window,
        x_is_mps=x_is_mps,
    )
    return stft_repr, context


def istft_roformer(module, stft_repr, context, length):
    b, n, _, t = stft_repr.shape
    stft_repr = stft_repr.reshape(b, n, context.freq_bins, context.channels, t).permute(0, 1, 3, 2, 4).reshape(
        b * n * context.channels,
        context.freq_bins,
        t
    )

    try:
        recon_audio = torch.istft(
            stft_repr,
            **module.stft_kwargs,
            window=context.stft_window,
            return_complex=False,
            length=length
        )
    except RuntimeError:
        recon_audio = torch.istft(
            stft_repr.cpu() if context.x_is_mps else stft_repr,
            **module.stft_kwargs,
            window=context.stft_window.cpu() if context.x_is_mps else context.stft_window,
            return_complex=False,
            length=length
        ).to(context.stft_window.device)

    recon_audio = recon_audio.reshape(context.batch, n, context.channels, recon_audio.shape[-1])

    if n == 1:
        return recon_audio[:, 0]

    return recon_audio


def forward_spectral_roformer(module, raw_audio, match_input_audio_length=True):
    stft_repr, context = stft_roformer(module, raw_audio)
    stft_repr = module._mask_stft_repr(stft_repr, context)
    length = context.audio_length if match_input_audio_length else None
    return istft_roformer(module, stft_repr, context, length)


def forward_bandsplit_roformer(module, raw_audio):
    return forward_spectral_roformer(module, raw_audio, match_input_audio_length=True)
