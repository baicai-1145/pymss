from torch.nn import Module
from .transformer import ConvFFN
from .common import (DEFAULT_FREQS_PER_BANDS, MaskEstimator, RMSNorm, RoformerRuntimeMixin, forward_bandsplit_roformer, forward_roformer_mask_core, ignore_roformer_training_kwargs, init_conformer_layers, init_roformer_band_modules, init_roformer_layers, init_roformer_runtime, init_roformer_shared_bias, init_roformer_stft, roformer_freqs_per_bands_with_complex, roformer_stft_freq_bins, roformer_transformer_kwargs)
class BSRoformer(RoformerRuntimeMixin, Module):
    # One class covers bs_roformer / bs_conformer: conformer=True swaps Transformer->Conformer layers
    # (identical layers.N.{0,1} state_dict layout, so checkpoints load either way).
    mask_estimator_cls = MaskEstimator
    def __init__(self, dim, *, depth, stereo=False, num_stems=1, time_transformer_depth=2, freq_transformer_depth=2,
                 freqs_per_bands=DEFAULT_FREQS_PER_BANDS, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0,
                 flash_attn=True, stft_n_fft=2048, stft_hop_length=512, stft_win_length=2048, stft_normalized=False,
                 stft_window_fn=None, mask_estimator_depth=2, mlp_expansion_factor=4, use_shared_bias=False,
                 zero_dc=False, skip_connection=False, conformer=False, ff_mult=4, conv_expansion_factor=2,
                 conv_kernel_size=31, norm_output=False, use_pope=False, conv_ffn_kernel=None, ff_moe_num_experts=None, ff_moe_top_k=2, **kwargs):
        super().__init__()
        ignore_roformer_training_kwargs(kwargs)
        if use_pope and conformer: raise ValueError("PoPE is only supported for BS-RoFormer")
        self.use_pope = bool(use_pope)
        init_roformer_runtime(self, stereo, num_stems, skip_connection=skip_connection)
        shared_qkv_bias, shared_out_bias = init_roformer_shared_bias(self, dim=dim, heads=heads, dim_head=dim_head, use_shared_bias=use_shared_bias)
        transformer_kwargs = roformer_transformer_kwargs(dim=dim, heads=heads, dim_head=dim_head, attn_dropout=attn_dropout, ff_dropout=ff_dropout, flash_attn=flash_attn, norm_output=norm_output, shared_qkv_bias=shared_qkv_bias, shared_out_bias=shared_out_bias)
        if ff_moe_num_experts:
            transformer_kwargs = dict(transformer_kwargs, ff_moe=dict(num_experts=int(ff_moe_num_experts), top_k=int(ff_moe_top_k)))
        if conformer: init_conformer_layers(self, depth=depth, time_conformer_depth=time_transformer_depth, freq_conformer_depth=freq_transformer_depth, dim_head=dim_head, transformer_kwargs=transformer_kwargs, ff_mult=ff_mult, conv_expansion_factor=conv_expansion_factor, conv_kernel_size=conv_kernel_size)
        else: init_roformer_layers(self, depth=depth, time_transformer_depth=time_transformer_depth, freq_transformer_depth=freq_transformer_depth, dim_head=dim_head, transformer_kwargs=transformer_kwargs, use_pope=self.use_pope)
        self.conv_ffn = ConvFFN(dim=dim, mult=ff_mult, k=int(conv_ffn_kernel), dropout=ff_dropout) if conv_ffn_kernel else None
        self.final_norm, self.zero_dc = RMSNorm(dim), zero_dc
        init_roformer_stft(self, stft_n_fft, stft_hop_length, stft_win_length, stft_normalized, stft_window_fn)
        freqs = roformer_stft_freq_bins(self, stft_win_length)
        init_roformer_band_modules(self, dim=dim, freqs_per_bands_with_complex=roformer_freqs_per_bands_with_complex(self, freqs_per_bands, freqs), num_stems=num_stems, mask_estimator_cls=self.mask_estimator_cls, mask_estimator_depth=mask_estimator_depth, mlp_expansion_factor=mlp_expansion_factor)
    def set_mps_model_backend(self, backend=None, compute_dtype=None):
        super().set_mps_model_backend(backend, compute_dtype)
        if self.use_pope: self.mps_model_backend = "torch"
    def _forward_mask_core(self, stft_repr): return forward_roformer_mask_core(self, stft_repr)
    def pop_moe_aux_loss(self):
        total = None
        for m in self.modules():
            if isinstance(m, __import__('pymss_core.modules.bs_roformer.transformer', fromlist=['MoEFFN']).MoEFFN):
                if m.aux_loss is not None:
                    total = m.aux_loss if total is None else total + m.aux_loss
                    m.aux_loss = None
        return total

    def forward(self, raw_audio):
        if self._use_mlx_full_forward(raw_audio):
            try:
                from .mlx_roformer import mlx_forward_roformer
                return mlx_forward_roformer(self, raw_audio, self.mps_model_compute_dtype)
            except Exception as exc:
                self._pymss_mlx_full_backend_error = repr(exc)
                self.mps_model_backend = "torch"
        return forward_bandsplit_roformer(self, raw_audio)
