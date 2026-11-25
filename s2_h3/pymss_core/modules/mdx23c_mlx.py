import torch
from .mdx23c_tfc_tdf_v3 import TFC_TDF, Downscale, Upscale
from .mlx_backend import (batch_norm, check_dtype, conv2d, generic_activation, generic_module_forward, group_norm, instance_norm2d, istft, linear_layer, mx_dtype, periodic_hann_window, reflect_pad_last, stft, to_mx, to_torch)
torch_to_mlx_input = to_mx
_linear_layer = linear_layer
def _subband_stft(module, raw_audio, dtype):
    import mlx.core as mx
    n_fft, hop, dim_f = int(module.stft.n_fft), int(module.stft.hop_length), int(module.stft.dim_f)
    window = periodic_hann_window(n_fft, dtype)
    spec = stft(raw_audio, n_fft, hop, window, dtype, pad_fn=reflect_pad_last)  # (B, C, F, T)
    channels, freq_bins, time_bins = spec.shape[-3:]
    ri = mx.stack((spec.real, spec.imag), axis=-3).reshape(*spec.shape[:-3], channels * 2, freq_bins, time_bins)
    return ri[..., :dim_f, :], {"audio_length": raw_audio.shape[-1], "n_fft": n_fft, "hop": hop, "window": window, "dtype": dtype}
def _subband_istft(module, x, context):
    import mlx.core as mx
    batch_dims = x.shape[:-3]
    _channels, freq_bins, time_bins = x.shape[-3:]
    n_fft = context["n_fft"]
    full_freq_bins = n_fft // 2 + 1
    if freq_bins < full_freq_bins: x = mx.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, full_freq_bins - freq_bins), (0, 0)])
    x = x.reshape(-1, 2, full_freq_bins, time_bins)
    spec = x[:, 0] + (1j * x[:, 1])  # (n, F, T)
    audio = istft(spec, context["window"], context["hop"], context["audio_length"], context["dtype"], n_fft=context["n_fft"])
    return audio.reshape(*batch_dims, 2, audio.shape[-1])
_norm = lambda module, x, dtype: (group_norm if isinstance(module, torch.nn.GroupNorm) else batch_norm if isinstance(module, torch.nn.BatchNorm2d) else instance_norm2d)(module, x, dtype)
_activation = generic_activation
def _module_forward(module, x, dtype):
    if isinstance(module, TFC_TDF): return _tfc_tdf(module, x, dtype)
    if isinstance(module, (Downscale, Upscale)): return _module_forward(module.conv, x, dtype)
    return generic_module_forward(module, x, dtype, _norm, extra=((torch.nn.InstanceNorm2d, instance_norm2d), (torch.nn.BatchNorm2d, batch_norm), (torch.nn.GroupNorm, group_norm)))
def _tfc_tdf(module, x, dtype):
    for block in module.blocks: shortcut = conv2d(block.shortcut, x, dtype); x = _module_forward(block.tfc1, x, dtype); x = _module_forward(block.tfc2, x + _module_forward(block.tdf, x, dtype), dtype) + shortcut
    return x
def _forward_core(module, x, dtype):
    import mlx.core as mx
    encoder_outputs = []
    for block in module.encoder_blocks: x = _tfc_tdf(block.tfc_tdf, x, dtype); encoder_outputs.append(x); x = _module_forward(block.downscale, x, dtype)
    x = _tfc_tdf(module.bottleneck_block, x, dtype)
    for block in module.decoder_blocks: x = _module_forward(block.upscale, x, dtype); x = _tfc_tdf(block.tfc_tdf, mx.concatenate((x, encoder_outputs.pop()), axis=1), dtype)
    return x
def mlx_forward_mdx23c_mx(module, raw_audio, dtype=torch.float16):
    import mlx.core as mx
    check_dtype(dtype, "MDX23C")
    dtype = mx_dtype(dtype)
    x, context = _subband_stft(module, raw_audio, dtype)
    n_sub = module.num_subbands
    mix = x = mx.reshape(x, (x.shape[0], x.shape[1] * n_sub, x.shape[2] // n_sub, x.shape[3]))  # cac -> cws
    first_conv_out = x = conv2d(module.first_conv, x, dtype)
    x = _forward_core(module, x.transpose(0, 1, 3, 2), dtype).transpose(0, 1, 3, 2)
    x = x * first_conv_out
    x = _module_forward(module.final_conv, mx.concatenate((mix, x), axis=1), dtype)
    batch, channels, freq_bins, time_bins = x.shape
    x = mx.reshape(x, (batch, channels // n_sub, freq_bins * n_sub, time_bins))  # cws -> cac
    if module.num_target_instruments > 1: x = x.reshape(batch, module.num_target_instruments, -1, freq_bins * n_sub, time_bins)
    return _subband_istft(module, x, context)
def mlx_forward_mdx23c(module, raw_audio, dtype=torch.float16): return to_torch(mlx_forward_mdx23c_mx(module, to_mx(raw_audio, dtype=dtype), dtype), raw_audio)