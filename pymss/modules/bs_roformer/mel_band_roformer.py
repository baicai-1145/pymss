import torch
from torch import nn
from torch.nn import Module

from typing import Callable, Optional

from einops import rearrange, reduce, repeat
from librosa import filters

from .common import (
    MaskEstimator,
    RoformerRuntimeMixin,
    forward_roformer_mask_core,
    forward_spectral_roformer,
    ignore_roformer_training_kwargs,
    init_roformer_band_modules,
    init_roformer_layers,
    init_roformer_runtime,
    init_roformer_stft,
)


# main class

class MelBandRoformer(RoformerRuntimeMixin, Module):

    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            num_bands=60,
            dim_head=64,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            flash_attn=True,
            sample_rate=44100,  # needed for mel filter bank from librosa
            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=1,
            match_input_audio_length=False,  # if True, pad output tensor to match length of input tensor
            mlp_expansion_factor=4,
            **kwargs,
    ):
        super().__init__()
        ignore_roformer_training_kwargs(kwargs)
        init_roformer_runtime(self, stereo, num_stems)

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
        )

        init_roformer_layers(
            self,
            depth=depth,
            time_transformer_depth=time_transformer_depth,
            freq_transformer_depth=freq_transformer_depth,
            dim_head=dim_head,
            transformer_kwargs=transformer_kwargs,
        )

        self.final_norm = nn.Identity()
        init_roformer_stft(self, stft_n_fft, stft_hop_length, stft_win_length, stft_normalized, stft_window_fn)

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_n_fft), return_complex=True).shape[1]

        # create mel filter bank
        # with librosa.filters.mel as in section 2 of paper

        mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        # for some reason, it doesn't include the first freq? just force a value for now

        mel_filter_bank[0][0] = 1.

        # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
        # so let's force a positive value

        mel_filter_bank[-1, -1] = 1.

        # binary as in paper (then estimated masks are averaged for overlapping regions)

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), 'all frequencies need to be covered by all bands for now'

        repeated_freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('freqs_per_band', freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

        self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False)
        self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False)
        self.register_buffer(
            'num_bands_per_channel_freq',
            num_bands_per_freq.repeat_interleave(self.audio_channels).view(1, 1, -1, 1),
            persistent=False
        )

        # band split and mask estimator

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())
        init_roformer_band_modules(
            self,
            dim=dim,
            freqs_per_bands_with_complex=freqs_per_bands_with_complex,
            num_stems=num_stems,
            mask_estimator_cls=MaskEstimator,
            mask_estimator_depth=mask_estimator_depth,
            mlp_expansion_factor=mlp_expansion_factor,
            mask_estimator_kwargs={'mlp_hidden_layers': mask_estimator_depth},
        )

        self.match_input_audio_length = match_input_audio_length

    def _forward_mask_core(self, selected_stft_repr):
        return forward_roformer_mask_core(self, selected_stft_repr)

    def _mask_stft_repr(self, stft_repr, context):
        x = stft_repr[torch.arange(context.batch, device=stft_repr.device)[..., None], self.freq_indices]
        self._warm_group_cache(x)
        masks = self._forward_mask_core(x)

        stft_repr = torch.view_as_complex(stft_repr.unsqueeze(1))
        masks = torch.view_as_complex(masks.contiguous()).to(dtype=stft_repr.dtype)
        num_stems = len(self.mask_estimators)
        scatter_indices = self.freq_indices.view(1, 1, -1, 1).expand(
            context.batch,
            num_stems,
            -1,
            stft_repr.shape[-1],
        )
        masks_summed = stft_repr.new_zeros(context.batch, num_stems, stft_repr.shape[2], stft_repr.shape[-1])
        masks_summed.scatter_add_(2, scatter_indices, masks)
        return stft_repr * (masks_summed / self.num_bands_per_channel_freq.clamp(min=1e-8))

    def forward(self, raw_audio):
        return forward_spectral_roformer(
            self,
            raw_audio,
            match_input_audio_length=self.match_input_audio_length,
        )
