from functools import partial
import torch
from torch import nn
from ..mlx_backend import MpsBackendMixin
from pymss_core.runtime_cache import _clear_cache_dicts
from .bands import BandSplit, MaskEstimator
from .conformer import Conformer
from .pope import PoPE
from .transformer import RMSNorm, Transformer  # noqa: F401 (RMSNorm re-export is part of the package surface)
DEFAULT_FREQS_PER_BANDS = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)
class SpectralContext(tuple):
    # (batch, channels, freq_bins, audio_length, stft_window, x_is_mps, x_is_npu) with named access
    def __new__(cls, batch, channels, freq_bins, audio_length, stft_window, x_is_mps, x_is_npu=False): return super().__new__(cls, (batch, channels, freq_bins, audio_length, stft_window, x_is_mps, x_is_npu))
    @property
    def batch(self): return self[0]
    @property
    def channels(self): return self[1]
    @property
    def freq_bins(self): return self[2]
    @property
    def audio_length(self): return self[3]
    @property
    def stft_window(self): return self[4]
    @property
    def x_is_mps(self): return self[5]
    @property
    def x_is_npu(self): return self[6]
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000): super().__init__(); freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)); self.freqs = nn.Parameter(freqs, requires_grad=False); self.cache = {}
    def clear_runtime_cache(self):
        self.cache.clear()
        getattr(self, "_pymss_cos_sin_cache", {}).clear()
    def _apply(self, fn, recurse=True):
        # Keep inverse frequencies precise when the surrounding network is cast.
        # Detach before super(): it may replace the Parameter storage in place.
        freqs = self.freqs.detach()
        result = super()._apply(fn, recurse=recurse)
        if self.freqs.dtype in (torch.float16, torch.bfloat16):
            self.freqs.data = freqs.to(device=self.freqs.device, dtype=torch.float32)
        self.clear_runtime_cache()
        return result
    def get_seq_pos(self, seq_len, device, dtype, offset=0): return torch.arange(seq_len, device=device, dtype=dtype) + offset
    def forward(self, t, cache_key=None):
        if cache_key in self.cache: return self.cache[cache_key]
        t = t() if callable(t) else t
        freqs = (t.to(self.freqs.dtype)[:, None] * self.freqs[None]).repeat_interleave(2, -1)
        if cache_key is not None: self.cache[cache_key] = freqs
        return freqs
def default(v, d): return v if v is not None else d
def mask_to_complex_shape(mask, complex_dim=2): b, n, t, fc = mask.shape; return mask.reshape(b, n, t, fc // complex_dim, complex_dim).permute(0, 1, 3, 2, 4)
TRAINING_LOSS_KWARGS = frozenset({"multi_stft_resolution_loss_weight", "multi_stft_resolutions_window_sizes", "multi_stft_hop_size", "multi_stft_normalized", "multi_stft_window_fn"})
REMOVED_ROFORMER_KWARGS = frozenset({"linear_transformer_depth", "linear_conformer_depth", "use_torch_checkpoint", "attention_layout", "dim_freqs_in", "sage_attention", "conv_dropout"})
def ignore_roformer_training_kwargs(kwargs):
    unexpected = set(kwargs) - TRAINING_LOSS_KWARGS - REMOVED_ROFORMER_KWARGS
    if unexpected: raise TypeError(f"unexpected RoFormer config keys: {sorted(unexpected)}")
def init_roformer_runtime(module, stereo, num_stems, skip_connection=False): module.stereo = stereo; module.audio_channels,module.num_stems,module.skip_connection = 2 if stereo else 1, num_stems, bool(skip_connection)
def init_roformer_shared_bias(module, dim, heads, dim_head, use_shared_bias):
    if not use_shared_bias: return None, None
    dim_inner = heads * dim_head
    module.linear_62_bias_0 = nn.Parameter(torch.ones(dim_inner * 3))
    module.linear_64_bias_0 = nn.Parameter(torch.ones(dim))
    return module.linear_62_bias_0, module.linear_64_bias_0
def roformer_transformer_kwargs(*, dim, heads, dim_head, attn_dropout, ff_dropout, flash_attn, norm_output=None, shared_qkv_bias=None, shared_out_bias=None): return dict(dim=dim, heads=heads, dim_head=dim_head, attn_dropout=attn_dropout, ff_dropout=ff_dropout, flash_attn=flash_attn, **{k: v for k, v in (("norm_output", norm_output), ("shared_qkv_bias", shared_qkv_bias), ("shared_out_bias", shared_out_bias)) if v is not None})
def init_roformer_layers(module, *, depth, time_transformer_depth, freq_transformer_depth, dim_head, transformer_kwargs, use_pope=False):
    # Share one embedding per axis across layers (RNG order frozen for seed compatibility).
    if use_pope:
        time_embed, freq_embed = PoPE(dim_head, heads=transformer_kwargs["heads"]), PoPE(dim_head, heads=transformer_kwargs["heads"])
    else:
        time_embed, freq_embed = RotaryEmbedding(dim=dim_head), RotaryEmbedding(dim=dim_head)
    embedding_key = "pope_embed" if use_pope else "rotary_embed"
    module.layers = nn.ModuleList([nn.ModuleList([ Transformer(depth=time_transformer_depth, **{embedding_key: time_embed}, **transformer_kwargs), Transformer(depth=freq_transformer_depth, **{embedding_key: freq_embed}, **transformer_kwargs)]) for _ in range(depth)])
def init_conformer_layers(module, *, depth, time_conformer_depth, freq_conformer_depth, dim_head, transformer_kwargs, ff_mult=4, conv_expansion_factor=2, conv_kernel_size=31):
    time_rotary, freq_rotary = RotaryEmbedding(dim=dim_head), RotaryEmbedding(dim=dim_head)
    ck = dict(ff_mult=ff_mult, conv_expansion_factor=conv_expansion_factor, conv_kernel_size=conv_kernel_size, **transformer_kwargs)
    module.layers = nn.ModuleList([nn.ModuleList([ Conformer(depth=time_conformer_depth, rotary_embed=time_rotary, **ck), Conformer(depth=freq_conformer_depth, rotary_embed=freq_rotary, **ck)]) for _ in range(depth)])
def init_roformer_stft(module, stft_n_fft, stft_hop_length, stft_win_length, stft_normalized, stft_window_fn): module.stft_kwargs = {"n_fft": stft_n_fft, "hop_length": stft_hop_length, "win_length": stft_win_length, "normalized": stft_normalized}; module.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length); module._stft_window_cache = {}
def roformer_stft_freq_bins(module, window_length):
    # The original training code computed this shape through torch.stft on a random probe tensor during model
    # construction. Preserve that RNG-consuming behavior so scratch init remains seed-compatible.
    return torch.stft(torch.randn(1, 4096), **module.stft_kwargs, window=torch.ones(window_length), return_complex=True).shape[1]
def roformer_freqs_per_bands_with_complex(module, freqs_per_bands, freqs):
    assert len(freqs_per_bands) > 1
    assert sum(freqs_per_bands) == freqs, (f"the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}")
    return tuple(2 * f * module.audio_channels for f in freqs_per_bands)
def init_roformer_band_modules(module, *, dim, freqs_per_bands_with_complex, num_stems, mask_estimator_cls, mask_estimator_depth, mlp_expansion_factor, mask_estimator_kwargs=None):
    module.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)
    module.mask_estimators = nn.ModuleList([ mask_estimator_cls(dim=dim, dim_inputs=freqs_per_bands_with_complex, depth=mask_estimator_depth, mlp_expansion_factor=mlp_expansion_factor, **(mask_estimator_kwargs or {})) for _ in range(num_stems)])
class RoformerRuntimeMixin(MpsBackendMixin):
    def clear_runtime_cache(self): _clear_cache_dicts(self, "_stft_window_cache")
    def mlx_forward_mx(self, raw_audio): from .mlx_roformer import mlx_forward_roformer_mx; return mlx_forward_roformer_mx(self, raw_audio, self.mps_model_compute_dtype)
    def stft_window(self, device):
        key = (device.type, device.index, torch.float32)
        window = self._stft_window_cache.get(key)
        if window is None or window.device != device:
            # Build the hann window in fp32 on the host and move it over: this
            # both avoids torch_npu's on-device int64/float cast bug and keeps
            # aten.hann_window out of torchair graphs (no AscendIR converter).
            if device.type == "npu":
                window = self.stft_window_fn(device="cpu", dtype=torch.float32).to(device)
            else:
                window = self.stft_window_fn(device=device)
            self._stft_window_cache[key] = window
        return window
    def _active_source_indices(self): indices = getattr(self, "_pymss_source_indices", None); return None if indices is None else tuple(int(index) for index in indices)
    def _active_mask_estimators(self): indices = self._active_source_indices(); return tuple(self.mask_estimators) if indices is None else tuple(self.mask_estimators[i] for i in indices)
    def _active_source_count(self): indices = self._active_source_indices(); return len(self.mask_estimators) if indices is None else len(indices)
    def _warm_group_cache(self, tensor):
        key = (tensor.device.type, tensor.device.index, tensor.dtype)
        if getattr(self, "_pymss_group_cache_warm_key", None) == key: return
        self.band_split.warm_group_cache(tensor.device, tensor.dtype)
        self._pymss_group_cache_warm_key = key
    def _estimate_masks(self, x):
        estimators = self._active_mask_estimators()
        if self._active_source_indices() is not None:
            packed = MaskEstimator.forward_packed_estimators(estimators, x)
            if packed is not None: return packed
            return torch.stack([fn(x) for fn in estimators], dim=1)
        use_packed = getattr(self, "_packed_mask_estimators_available", None)
        if use_packed is not False:
            packed = MaskEstimator.forward_packed_estimators(estimators, x)
            if packed is not None: self._packed_mask_estimators_available = True; return packed
            self._packed_mask_estimators_available = False
        return torch.stack([fn(x) for fn in estimators], dim=1)
    def _mask_stft_repr(self, stft_repr, context):
        self._warm_group_cache(stft_repr)
        mask = self._forward_mask_core(stft_repr)
        if mask.dtype != stft_repr.dtype:
            # torch_npu cannot copy between complex dtypes (complex32 ->
            # complex64); cast the underlying real tensor instead.
            mask = mask.to(dtype=stft_repr.dtype)
        stft_repr = torch.view_as_complex(stft_repr.unsqueeze(1))
        return stft_repr * torch.view_as_complex(mask).to(dtype=stft_repr.dtype)
def forward_roformer_mask_core(module, stft_repr):
    b, fs, model_t, complex_dim = stft_repr.shape
    # STFT stays fp32 for complex64 reconstruction. Explicit low-precision model
    # conversion must cast features at the network boundary, without autocast.
    features = stft_repr.permute(0, 2, 1, 3).reshape(b, model_t, fs * complex_dim)
    weights = next(module.band_split.parameters(), None) if isinstance(module.band_split, nn.Module) else None
    if weights is not None:
        features = features.to(dtype=weights.dtype)
    x = module.band_split(features)
    residual_store = [] if getattr(module, "skip_connection", False) else None
    for time_transformer, freq_transformer in module.layers:
        if residual_store is not None:
            for residual in residual_store: x = x + residual
        b, t, f, d = x.shape
        x = time_transformer(x.permute(0, 2, 1, 3).reshape(b * f, t, d)).reshape(b, f, t, d).permute(0, 2, 1, 3)
        x = freq_transformer(x.reshape(b * t, f, d)).reshape(b, t, f, d)
        if getattr(module, "conv_ffn", None) is not None: x = module.conv_ffn(x) + x
        if residual_store is not None: residual_store.append(x)
    return mask_to_complex_shape(module._estimate_masks(module.final_norm(x)), complex_dim=2)
def stft_roformer(module, raw_audio):
    from pymss_core.npu_fft import stft_npu, torch_stft_blocked
    device, x_is_mps = raw_audio.device, raw_audio.device.type == "mps"
    x_is_npu = torch_stft_blocked(raw_audio.device)
    if raw_audio.ndim == 2: raw_audio = raw_audio.unsqueeze(1)
    batch, audio_channels, audio_length = raw_audio.shape
    assert (not module.stereo and audio_channels == 1) or (module.stereo and audio_channels == 2), ("stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). " "also need to be False if mono (channel dimension of 1)")
    stft_window = module.stft_window(device)
    flat = raw_audio.reshape(batch * audio_channels, audio_length)
    if x_is_npu:
        stft_repr = stft_npu(flat.float(), **module.stft_kwargs, window=stft_window, return_complex=True)
    else:
        try:
            stft_repr = torch.stft(flat, **module.stft_kwargs, window=stft_window, return_complex=True)
        except RuntimeError:  # older MPS torch.stft: fall back to CPU
            stft_repr = torch.stft(flat.cpu() if x_is_mps else flat, **module.stft_kwargs, window=stft_window.cpu() if x_is_mps else stft_window, return_complex=True).to(device)
    stft_repr = torch.view_as_real(stft_repr).reshape(batch, audio_channels, -1, stft_repr.shape[-1], 2)
    b, s, f, t, c = stft_repr.shape
    return (stft_repr.permute(0, 2, 1, 3, 4).reshape(b, f * s, t, c), SpectralContext(batch, audio_channels, f, audio_length, stft_window, x_is_mps, x_is_npu))
def istft_roformer(module, stft_repr, context, length):
    from pymss_core.npu_fft import istft_npu
    # Input is complex (b, n, f*channels, t) — the product of stft_repr * mask.
    # torch_npu runs complex64 transposes on the AI_CPU (~6ms each); rearrange
    # channels in the REAL domain on the vector cores instead.
    b, n, fs, t = stft_repr.shape
    real = (torch.view_as_real(stft_repr)
            .reshape(b, n, context.freq_bins, context.channels, t, 2)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b * n * context.channels, context.freq_bins, t, 2).contiguous())
    stft_repr = torch.view_as_complex(real)
    if getattr(module, "zero_dc", False):
        # In-place assignment on a complex tensor is not usable on every vendor
        # (BIREN: "devToAccess -1 is not a device"), so the DC bin is replaced
        # functionally instead.
        planes = torch.view_as_real(stft_repr)
        planes = torch.cat((torch.zeros_like(planes[:, :1]), planes[:, 1:]), dim=1)
        stft_repr = torch.view_as_complex(planes.contiguous())
    if context.x_is_npu:
        recon_audio = istft_npu(stft_repr, **module.stft_kwargs, window=context.stft_window, return_complex=False, length=length)
    else:
        try:
            recon_audio = torch.istft(stft_repr, **module.stft_kwargs, window=context.stft_window, return_complex=False, length=length)
        except RuntimeError:  # older MPS torch.istft: fall back to CPU
            recon_audio = torch.istft(stft_repr.cpu() if context.x_is_mps else stft_repr, **module.stft_kwargs, window=context.stft_window.cpu() if context.x_is_mps else context.stft_window, return_complex=False, length=length).to(context.stft_window.device)
    recon_audio = recon_audio.reshape(context.batch, n, context.channels, recon_audio.shape[-1])
    return recon_audio[:, 0] if n == 1 else recon_audio
def forward_spectral_roformer(module, raw_audio, match_input_audio_length=True): stft_repr, context = stft_roformer(module, raw_audio); return istft_roformer(module, module._mask_stft_repr(stft_repr, context), context, context.audio_length if match_input_audio_length else None)
def forward_bandsplit_roformer(module, raw_audio): return forward_spectral_roformer(module, raw_audio, match_input_audio_length=True)
