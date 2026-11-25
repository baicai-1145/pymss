import torch
from .look2hear.apollo import ICB, BSNet, ConvActNorm1d, RMSNorm
from .mlx_backend import check_dtype, conv1d, istft, mx_dtype, param, reflect_pad_last, silu, stft, to_mx, to_torch
torch_to_mlx_input = to_mx
def _stft(module, raw_audio, dtype): window = to_mx(module.window, torch.float32).astype(dtype); context = {"length": raw_audio.shape[-1], "n_fft": module.win, "hop": module.stride, "window": window, "dtype": dtype}; return stft(raw_audio, module.win, module.stride, window, dtype, pad_fn=reflect_pad_last), context
def _rms_norm(module, x, dtype): import mlx.core as mx; batch, channels, frames = x.shape; groups = int(module.groups); y = x.astype(mx.float32).reshape(batch, groups, channels // groups, frames); y = y * mx.rsqrt(mx.mean(mx.square(y), axis=2, keepdims=True) + module.eps); y = y.reshape(batch, channels, frames).astype(x.dtype); return y * param(module, "weight", module.weight, dtype).reshape(1, -1, 1)
def _module_forward(module, x, dtype):
    import mlx.core as mx
    if isinstance(module, torch.nn.Sequential):
        for child in module: x = _module_forward(child, x, dtype)
        return x
    if isinstance(module, torch.nn.Conv1d): return conv1d(module, x, dtype)
    if isinstance(module, RMSNorm): return _rms_norm(module, x, dtype)
    if isinstance(module, torch.nn.SiLU): return silu(x)
    if isinstance(module, torch.nn.GLU): a, b = mx.split(x, 2, axis=module.dim); return a * mx.sigmoid(b)
    if isinstance(module, ConvActNorm1d): return _conv_act_norm(module, x, dtype)
    if isinstance(module, ICB): return _module_forward(module.blocks, x, dtype)
    if isinstance(module, BSNet): return _bsnet(module, x, dtype)
    raise TypeError(f"unsupported Apollo layer for MLX full backend: {type(module).__name__}")
def _conv_act_norm(module, x, dtype):
    y = conv1d(module.conv[0], x, dtype)
    y = _rms_norm(module.conv[1], y, dtype)
    y = silu(conv1d(module.conv[2], y, dtype))
    y = conv1d(module.conv[4], y, dtype)
    if module.causal: y = y[..., :-module.kernel + 1]
    return x + y
def _apply_rope(module, x, dtype): import mlx.core as mx; seq_len = x.shape[-2]; cos = to_mx(module.cos_freq[:seq_len, 0::2], dtype).reshape(1, 1, seq_len, -1); sin = to_mx(module.sin_freq[:seq_len, 0::2], dtype).reshape(1, 1, seq_len, -1); even, odd = x[..., 0::2], x[..., 1::2]; return mx.stack((even * cos - odd * sin, odd * cos + even * sin), axis=-1).reshape(x.shape)
def _roformer(module, x, dtype):
    import mlx.core as mx
    batch, _, frames = x.shape
    qkv = conv1d(module.weight, _rms_norm(module.input_norm, x, dtype), dtype)
    qkv = qkv.reshape(batch, module.num_head, module.hidden_size * 3, frames).transpose(0, 1, 3, 2)
    q, k, v = mx.split(qkv, 3, axis=-1)
    attn = mx.fast.scaled_dot_product_attention(_apply_rope(module, q, dtype), _apply_rope(module, k, dtype), v, scale=module.hidden_size**-0.5, mask=None)
    out = conv1d(module.output, attn.transpose(0, 1, 3, 2).reshape(batch, -1, frames), dtype) + x
    hidden = silu(conv1d(module.MLP[1], _rms_norm(module.MLP[0], out, dtype), dtype))
    gate, z = mx.split(hidden, 2, axis=1)
    return out + conv1d(module.MLP_output, silu(gate) * z, dtype)
def _bsnet(module, x, dtype): batch, bands, channels, frames = x.shape; band = _roformer(module.band_net, x.transpose(0, 3, 2, 1).reshape(batch * frames, channels, bands), dtype); seq = _module_forward(module.seq_net, band.reshape(batch, frames, channels, bands).transpose(0, 3, 2, 1) .reshape(batch * bands, channels, frames), dtype); return seq.reshape(batch, bands, channels, frames)
def _feature_extractor(module, raw_audio, dtype):
    import mlx.core as mx
    batch, channels, samples = raw_audio.shape
    spec, _ = _stft(module, raw_audio.reshape(batch * channels, samples), dtype)
    features, band_index = [], 0
    for width, bn in zip(module.band_width, module.BN): sub = spec[:, band_index : band_index + width]; power = mx.sqrt(mx.sum(mx.square(sub.real) + mx.square(sub.imag), axis=1, keepdims=True) + module.eps); inp = mx.concatenate(((sub / power).real, (sub / power).imag, mx.log(power)), axis=1); features.append(_module_forward(bn, inp.astype(dtype), dtype)); band_index += width
    return mx.stack(features, axis=1), spec
def _estimate_spec(module, feature, batch_channels, dtype):
    import mlx.core as mx
    specs = []
    for band_feature, output, width in zip(mx.split(feature, feature.shape[1], axis=1), module.output, module.band_width): ri = _module_forward(output, band_feature[:, 0], dtype).reshape(batch_channels, 2, width, -1); specs.append(ri[:, 0] + (1j * ri[:, 1]))
    return mx.concatenate(specs, axis=1)
def mlx_forward_apollo_mx(module, raw_audio, dtype=torch.float16):
    check_dtype(dtype, "Apollo")
    dtype = mx_dtype(dtype)
    batch, channels, samples = raw_audio.shape
    feature, _ = _feature_extractor(module, raw_audio, dtype)
    for block in module.net: feature = _bsnet(block, feature, dtype)
    est_spec = _estimate_spec(module, feature, batch * channels, dtype)
    return istft(est_spec, to_mx(module.window, torch.float32).astype(raw_audio.dtype), module.stride, samples, raw_audio.dtype, n_fft=module.win).reshape(batch, channels, -1)
def mlx_forward_apollo(module, raw_audio, dtype=torch.float16): return to_torch(mlx_forward_apollo_mx(module, to_mx(raw_audio, dtype=dtype), dtype), raw_audio)