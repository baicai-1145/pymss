import math
import torch
from .mlx_backend import (check_dtype, conv2d, conv_transpose2d, gelu, generic_module_forward, glu, group_norm, istft, linear_layer, mx_dtype, rnn_forward, stft, to_mx, to_torch)
from .scnet.scnet import Swish
torch_to_mlx_input = to_mx
def _stft_scnet(module, raw_audio, dtype):
    cfg = module.stft_config
    n_fft, hop, win_length = int(cfg["n_fft"]), int(cfg["hop_length"]), int(cfg["win_length"])
    normalized, center = bool(cfg.get("normalized", False)), bool(cfg.get("center", True))
    window = mx_pad_window(win_length, n_fft, dtype)
    spec = stft(raw_audio, n_fft, hop, window, dtype, center=center, normalized=normalized, pad_fn=None)
    return spec, {"n_fft": n_fft, "hop": hop, "window": window, "normalized": normalized, "center": center, "dtype": dtype}
def mx_pad_window(win_length, n_fft, dtype):
    import mlx.core as mx
    window = mx.ones((win_length,), dtype=dtype)
    if win_length < n_fft: left = (n_fft - win_length) // 2; window = mx.pad(window, [(left, n_fft - win_length - left)])
    return window
def _istft_scnet(module, spec, context, length): return istft(spec, context["window"], context["hop"], length, context["dtype"], n_fft=context["n_fft"], center=context["center"], normalized=context["normalized"])
_linear_layer = linear_layer
def _seq(module, x, dtype):
    for child in module: x = _module_forward(child, x, dtype)
    return x
_module_forward = lambda module, x, dtype: generic_module_forward(module, x, dtype, group_norm, swish_cls=Swish)
def _sdlayer(module, x, dtype):
    import mlx.core as mx
    fr = x.shape[2]
    low, mid = math.ceil(fr * module.SR_low), math.ceil(fr * (module.SR_low + module.SR_mid))
    outputs, original_lengths = [], []
    for conv, stride, kernel, (start, end) in zip(module.convs, module.strides, module.kernels, [(0, low), (low, mid), (mid, fr)]): extracted = x[:, :, start:end, :]; original_lengths.append(end - start); total_padding = kernel - stride if stride == 1 else (stride - extracted.shape[2] % stride) % stride; pad_left = total_padding // 2; outputs.append(conv2d(conv, mx.pad(extracted, [(0, 0), (0, 0), (pad_left, total_padding - pad_left), (0, 0)]), dtype))
    return outputs, original_lengths
def _sdblock(module, x, dtype):
    import mlx.core as mx
    bands, original_lengths = _sdlayer(module.SDlayer, x, dtype)
    outs = []
    for conv, band in zip(module.conv_modules, bands): b, c, f, t = band.shape; out = _convolution_module(conv, band.transpose(0, 2, 1, 3).reshape(b * f, c, t), dtype); outs.append(gelu(out.reshape(b, f, c, t).transpose(0, 2, 1, 3)))
    lengths = [band.shape[-2] for band in outs]
    full_band = mx.concatenate(outs, axis=2)
    return conv2d(module.globalconv, full_band, dtype), full_band, lengths, original_lengths
def _convolution_module(module, x, dtype):
    for layer in module.layers: x = x + _seq(layer, x, dtype)
    return x
def _dual_path_rnn(module, x, dtype):
    b, c, f, t = x.shape
    y = group_norm(module.norm_layers[0], x, dtype).transpose(0, 3, 2, 1).reshape(b * t, f, c)
    y = rnn_forward(module.lstm_layers[0], y, dtype)
    y = _linear_layer(module.linear_layers[0], y, dtype)
    x = y.reshape(b, t, f, c).transpose(0, 3, 2, 1) + x
    y = group_norm(module.norm_layers[1], x, dtype).transpose(0, 2, 3, 1).reshape(b * f, t, c)
    y = rnn_forward(module.lstm_layers[1], y, dtype)
    y = _linear_layer(module.linear_layers[1], y, dtype)
    return y.reshape(b, f, t, c).transpose(0, 3, 1, 2) + x
def _feature_conversion(module, x):
    import mlx.core as mx
    x = x.astype(mx.float32)
    if module.inverse: half = module.channels // 2; return mx.fft.irfft(x[:, :half] + 1j * x[:, half:], n=(x.shape[3] - 1) * 2, axis=3, norm='ortho')
    x = mx.fft.rfft(x, axis=3, norm="ortho")
    return mx.concatenate((x.real, x.imag), axis=1)
def _separation_net(module, x, dtype):
    for dp_module, feature_conversion in zip(module.dp_modules, module.feature_conversion): x = _feature_conversion(feature_conversion, _dual_path_rnn(dp_module, x, dtype))
    return x
def _fusion_layer(module, x, skip, dtype):
    import mlx.core as mx
    if skip is not None: x = x + skip
    return glu(conv2d(module.conv, mx.concatenate((x, x), axis=1), dtype), axis=1)
def _sulayer(module, x, lengths, origin_lengths, dtype):
    import mlx.core as mx
    ranges = [(0, lengths[0]), (lengths[0], lengths[0] + lengths[1]), (lengths[0] + lengths[1], None)]
    outs = []
    for idx, (convtr, (start, end)) in enumerate(zip(module.convtrs, ranges)): out = conv_transpose2d(convtr, x[:, :, start:end, :], dtype); dist = abs(origin_lengths[idx] - out.shape[2]) // 2; outs.append(out[:, :, dist : dist + origin_lengths[idx], :])
    return mx.concatenate(outs, axis=2)
def mlx_forward_scnet_mx(module, raw_audio, dtype=torch.float16):
    import mlx.core as mx
    check_dtype(dtype, "SCNet")
    dtype = mx_dtype(dtype)
    x = raw_audio.astype(dtype)
    batch, padding = x.shape[0], module.hop_length - x.shape[-1] % module.hop_length
    if (x.shape[-1] + padding) // module.hop_length % 2 == 0: padding += module.hop_length
    x = mx.pad(x, [(0, 0), (0, 0), (0, padding)])
    length = x.shape[-1]
    spec, context = _stft_scnet(module, x.reshape(-1, length), dtype)
    ri = mx.stack((spec.real, spec.imag), axis=-1)
    x = ri.transpose(0, 3, 1, 2).reshape(ri.shape[0] // module.audio_channels, ri.shape[3] * module.audio_channels, ri.shape[1], ri.shape[2])
    _, _, freq_bins, time_bins = x.shape
    saved = []
    for sd_layer in module.encoder: x, skip, lengths, original_lengths = _sdblock(sd_layer, x, dtype); saved.append((skip, lengths, original_lengths))
    x = _separation_net(module.separation_net, x, dtype)
    for fusion_layer, su_layer in module.decoder: skip, lengths, original_lengths = saved.pop(); x = _sulayer(su_layer, _fusion_layer(fusion_layer, x, skip, dtype), lengths, original_lengths, dtype)
    x = x.reshape(-1, 2, freq_bins, time_bins)
    spec_out = x[:, 0] + (1j * x[:, 1])
    audio = _istft_scnet(module, spec_out, context, length)
    audio = audio.reshape(batch, len(module.sources), module.audio_channels, -1)
    return audio[:, :, :, :-padding] if padding > 0 else audio
def mlx_forward_scnet(module, raw_audio, dtype=torch.float16): return to_torch(mlx_forward_scnet_mx(module, to_mx(raw_audio, dtype=dtype), dtype), raw_audio)