import math
import torch
from .demucs_local import LayerScale, MyGroupNorm
from .mlx_backend import (check_dtype, conv1d, gelu, generic_module_forward, glu, group_norm, istft, layer_norm, linear, linear_layer, mx_dtype, pad_last, param, periodic_hann_window, relu, stft, to_mx, to_torch)
torch_to_mlx_input = to_mx
def _pad1d(x, paddings, mode="constant", value=0.0): return pad_last(x, paddings[0], paddings[1], mode=mode, value=value, extend=True)
def _spectro(x, n_fft, hop, dtype): return stft(x, n_fft, hop, periodic_hann_window(n_fft, dtype), dtype, normalized=True)
def _ispectro(z, hop, length, dtype): n_fft = 2 * z.shape[-2] - 2; return istft(z, periodic_hann_window(n_fft, dtype), hop, length, dtype, normalized=True)
def _demucs_spec(module, x, dtype): hop = module.hop_length; le = math.ceil(x.shape[-1] / hop); pad = hop // 2 * 3; x = _pad1d(x, (pad, pad + le * hop - x.shape[-1]), mode="reflect"); return _spectro(x, module.nfft, hop, dtype)[..., :-1, 2 : 2 + le]
def _demucs_ispec(module, z, length, scale, dtype): import mlx.core as mx; hop = module.hop_length // (4**scale); z = mx.pad(z, [(0, 0)] * (z.ndim - 2) + [(0, 1), (2, 2)]); pad = hop // 2 * 3; le = hop * math.ceil(length / hop) + 2 * pad; return _ispectro(z, hop, le, dtype)[..., pad : pad + length]
_linear_layer = linear_layer
def _my_group_norm(module, x, dtype): return group_norm(module, x.transpose(0, 2, 1), dtype).transpose(0, 2, 1)
def _norm(module, x, dtype):
    if isinstance(module, MyGroupNorm): return _my_group_norm(module, x, dtype)
    if isinstance(module, torch.nn.GroupNorm): return group_norm(module, x, dtype)
    if isinstance(module, torch.nn.LayerNorm): return layer_norm(module, x, dtype)
    if isinstance(module, torch.nn.Identity): return x
    raise TypeError(f"unsupported Demucs norm for MLX full backend: {type(module).__name__}")
def _activation(module, x):
    name = getattr(module, "__name__", None)
    if name == "gelu" or isinstance(module, torch.nn.GELU): return gelu(x)
    if name == "relu" or isinstance(module, torch.nn.ReLU): return relu(x)
    if isinstance(module, torch.nn.Identity): return x
    raise TypeError(f"unsupported Demucs activation for MLX full backend: {type(module).__name__}")
def _layer_scale(module, x, dtype): scale = param(module, "scale", module.scale, dtype); return scale * x if module.channel_last else scale[:, None] * x
def _module_forward(module, x, dtype):
    # F.gelu/F.relu raw functions (stored on CrossTransformerEncoderLayer) + MyGroupNorm/LayerScale are family-specific
    if isinstance(module, (torch.nn.GELU, torch.nn.ReLU)) or module.__class__ in (torch.nn.GELU, torch.nn.ReLU): return _activation(module, x)
    if isinstance(module, MyGroupNorm): return _my_group_norm(module, x, dtype)
    if isinstance(module, LayerScale): return _layer_scale(module, x, dtype)
    if isinstance(module, torch.nn.GroupNorm): return group_norm(module, x, dtype)
    if isinstance(module, torch.nn.LayerNorm): return layer_norm(module, x, dtype)
    if isinstance(module, torch.nn.Identity): return x
    return generic_module_forward(module, x, dtype, _norm, extra=((LayerScale, _layer_scale),))
def _seq(module, x, dtype):
    for child in module: x = _module_forward(child, x, dtype)
    return x
def _dconv(module, x, dtype):
    for layer in module.layers: x = x + _seq(layer, x, dtype)
    return x
def _freq_dconv(module, y, dtype): b, c, fr, t = y.shape; return _dconv(module.dconv, y.transpose(0, 2, 1, 3).reshape(-1, c, t), dtype).reshape(b, fr, c, t).transpose(0, 2, 1, 3)
def _henc_layer(module, x, inject, dtype):
    import mlx.core as mx
    if not module.freq and x.ndim == 4: x = x.reshape(x.shape[0], -1, x.shape[-1])
    if not module.freq and x.shape[-1] % module.stride: x = mx.pad(x, [(0, 0), (0, 0), (0, module.stride - x.shape[-1] % module.stride)])
    y = _module_forward(module.conv, x, dtype)
    if module.empty: return y
    if inject is not None: y = y + (inject[:, :, None] if inject.ndim == 3 and y.ndim == 4 else inject)
    y = gelu(_norm(module.norm1, y, dtype))
    if module.dconv: y = _freq_dconv(module, y, dtype) if module.freq else _dconv(module.dconv, y, dtype)
    return glu(_norm(module.norm2, _module_forward(module.rewrite, y, dtype), dtype), axis=1) if module.rewrite else y
def _hdec_layer(module, x, skip, length, dtype):
    if module.freq and x.ndim == 3: x = x.reshape(x.shape[0], module.chin, -1, x.shape[-1])
    if module.empty:
        y = x
    else:
        y = glu(_norm(module.norm1, _module_forward(module.rewrite, x + skip, dtype), dtype), axis=1) if module.rewrite else x + skip
        if module.dconv: y = _freq_dconv(module, y, dtype) if module.freq else _dconv(module.dconv, y, dtype)
    z = _norm(module.norm2, _module_forward(module.conv_tr, y, dtype), dtype)
    if module.freq and module.pad:
        z = z[..., module.pad : -module.pad, :]
    elif not module.freq: z = z[..., module.pad:module.pad + length]
    return (z if module.last else gelu(z)), y
def _create_2d_sin_embedding(d_model, height, width, dtype, max_period=10000):
    import mlx.core as mx
    half = d_model // 2
    div = mx.exp(mx.arange(0.0, half, 2) * -(math.log(max_period) / half))
    pos_w, pos_h = mx.arange(0.0, width).reshape(-1, 1), mx.arange(0.0, height).reshape(-1, 1)
    pe = mx.zeros((d_model, height, width), dtype=mx.float32)
    for sl, val in ((slice(0, half, 2), mx.sin(pos_w * div).transpose(1, 0)[:, None, :]), (slice(1, half, 2), mx.cos(pos_w * div).transpose(1, 0)[:, None, :]), (slice(half, None, 2), mx.sin(pos_h * div).transpose(1, 0)[:, :, None]), (slice(half + 1, None, 2), mx.cos(pos_h * div).transpose(1, 0)[:, :, None])): pe = pe.at[sl].add(val)
    return pe[None].astype(dtype)
def _create_sin_embedding(length, dim, dtype, max_period=10000): import mlx.core as mx; half = dim // 2; phase = mx.arange(length, dtype=mx.float32).reshape(-1, 1, 1) / (max_period ** (mx.arange(half, dtype=mx.float32).reshape(1, 1, -1) / (half - 1))); return mx.concatenate((mx.cos(phase), mx.sin(phase)), axis=-1).astype(dtype)
def _attention_out(mha, out, q_in, dtype): return _linear_layer(mha.out_proj, out.transpose(0, 2, 1, 3).reshape(q_in.shape), dtype)
def _split_heads(q, heads): return q.reshape(q.shape[0], q.shape[1], heads, -1).transpose(0, 2, 1, 3)
def _self_attention(mha, x, dtype): import mlx.core as mx; qkv = linear(x, param(mha, "in_proj_weight", mha.in_proj_weight, dtype), param(mha, "in_proj_bias", mha.in_proj_bias, dtype)); q, k, v = (mx.split(qkv, 3, axis=-1)); head_dim = q.shape[-1] // mha.num_heads; q, k, v = _split_heads(q, mha.num_heads), _split_heads(k, mha.num_heads), _split_heads(v, mha.num_heads); out = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5); return _attention_out(mha, out, x, dtype)
def _cross_attention(mha, q_in, k_in, dtype):
    import mlx.core as mx
    w = param(mha, "in_proj_weight", mha.in_proj_weight, dtype)
    b = param(mha, "in_proj_bias", mha.in_proj_bias, dtype)
    qw, kw, vw = mx.split(w, 3, axis=0)
    qb, kb, vb = mx.split(b, 3, axis=0)
    dim = q_in.shape[-1]
    head_dim = dim // mha.num_heads
    q = _split_heads(linear(q_in, qw, qb), mha.num_heads)
    k = _split_heads(linear(k_in, kw, kb), mha.num_heads)
    v = _split_heads(linear(k_in, vw, vb), mha.num_heads)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5)
    return _attention_out(mha, out, q_in, dtype)
def _ffn(module, x, dtype): return _module_forward(module.linear2, _activation(module.activation, _module_forward(module.linear1, x, dtype)), dtype)
def _transformer_encoder_layer(module, x, dtype):
    if module.norm_first: x = x + _layer_scale(module.gamma_1, _self_attention(module.self_attn, _norm(module.norm1, x, dtype), dtype), dtype); x = x + _layer_scale(module.gamma_2, _ffn(module, _norm(module.norm2, x, dtype), dtype), dtype); return _norm(module.norm_out, x, dtype) if module.norm_out else x
    x = _norm(module.norm1, x + _layer_scale(module.gamma_1, _self_attention(module.self_attn, x, dtype), dtype), dtype)
    return _norm(module.norm2, x + _layer_scale(module.gamma_2, _ffn(module, x, dtype), dtype), dtype)
def _cross_transformer_layer(module, q, k, dtype):
    if module.norm_first:
        attn = _cross_attention(module.cross_attn, _norm(module.norm1, q, dtype), _norm(module.norm2, k, dtype), dtype)
        x = q + _layer_scale(module.gamma_1, attn, dtype)
        x = x + _layer_scale(module.gamma_2, _ffn(module, _norm(module.norm3, x, dtype), dtype), dtype)
        return _norm(module.norm_out, x, dtype) if module.norm_out else x
    x = _norm(module.norm1, q + _layer_scale(module.gamma_1, _cross_attention(module.cross_attn, q, k, dtype), dtype), dtype)
    return _norm(module.norm2, x + _layer_scale(module.gamma_2, _ffn(module, x, dtype), dtype), dtype)
def _cross_transformer(module, x, xt, dtype):
    b, c, fr, t1 = x.shape
    pos = _create_2d_sin_embedding(c, fr, t1, x.dtype, module.max_period).transpose(0, 3, 2, 1).reshape(1, t1 * fr, c)
    x = _norm(module.norm_in, x.transpose(0, 3, 2, 1).reshape(b, t1 * fr, c), dtype) + module.weight_pos_embed * pos
    b, c, t2 = xt.shape
    xt_pos = _create_sin_embedding(t2, c, xt.dtype, module.max_period).transpose(1, 0, 2)
    xt = _norm(module.norm_in_t, xt.transpose(0, 2, 1), dtype) + module.weight_pos_embed * xt_pos
    for idx in range(module.num_layers):
        if idx % 2 == module.classic_parity:
            x, xt = _transformer_encoder_layer(module.layers[idx], x, dtype), _transformer_encoder_layer(module.layers_t[idx], xt, dtype)
        else:
            old_x = x
            x = _cross_transformer_layer(module.layers[idx], x, xt, dtype)
            xt = _cross_transformer_layer(module.layers_t[idx], xt, old_x, dtype)
    return x.reshape(b, t1, fr, c).transpose(0, 3, 2, 1), xt.transpose(0, 2, 1)
def _std(x, axes, keepdims): import math; import mlx.core as mx; mean = mx.mean(x, axis=axes, keepdims=True); return mx.sqrt(mx.sum(mx.square(x - mean), axis=axes, keepdims=keepdims) / max(1, math.prod(x.shape[a] for a in axes) - 1))
def _validate_supported(module):
    if module.num_subbands != 1 or not module.cac or module.wiener_iters != 0 or module.end_iters != 0: raise TypeError("MLX full HTDemucs supports num_subbands=1, cac=True, wiener_iters=end_iters=0")
    if any(layer.__class__.__name__ == "MultiWrap" for layer in list(module.encoder) + list(module.decoder)): raise TypeError("MLX full HTDemucs does not support MultiWrap/multi_freqs yet")
    if module.crosstransformer is not None and module.crosstransformer.emb != "sin": raise TypeError("MLX full HTDemucs supports sinusoidal transformer embeddings only")
def mlx_forward_demucs_mx(module, mix, dtype=torch.float16):
    import mlx.core as mx
    _validate_supported(module)
    check_dtype(dtype, "HTDemucs")
    dtype = mx_dtype(dtype)
    mix, length, length_pre_pad = mix.astype(dtype), mix.shape[-1], None
    if module.use_train_segment:
        training_length = int(module.segment * module.samplerate)
        if mix.shape[-1] < training_length: length_pre_pad = mix.shape[-1]; mix = mx.pad(mix, [(0, 0), (0, 0), (0, training_length - length_pre_pad)])
    z = _demucs_spec(module, mix, dtype)
    b, c, fr, t = z.shape
    x = mx.stack((z.real, z.imag), axis=2).reshape(b, c * 2, fr, t)
    f_query = x.shape[2]
    mean, std = mx.mean(x, axis=(1, 2, 3), keepdims=True), _std(x, (1, 2, 3), keepdims=True)
    x = (x - mean) / (1e-5 + std)
    meant, stdt = mx.mean(mix, axis=(1, 2), keepdims=True), _std(mix, (1, 2), keepdims=True)
    xt = (mix - meant) / (1e-5 + stdt)
    saved, saved_t, lengths_t = [], [], []
    for idx, encode in enumerate(module.encoder):
        skip_length = x.shape[-1]
        inject = None
        if idx < len(module.tencoder):
            lengths_t.append(xt.shape[-1])
            tenc = module.tencoder[idx]
            xt = _henc_layer(tenc, xt, None, dtype)
            if not tenc.empty: saved_t.append(xt)
            else: inject = xt
        x = _henc_layer(encode, x, inject, dtype)
        if idx == 0 and module.freq_emb is not None: weight = param(module.freq_emb.embedding, 'weight', module.freq_emb.embedding.weight, dtype) * module.freq_emb.scale; x = x + module.freq_emb_scale * weight[:x.shape[-2]].transpose(1, 0)[None, :, :, None]
        saved.append((x, skip_length))
    if module.crosstransformer:
        if module.bottom_channels: b, c, f, t = x.shape; x = conv1d(module.channel_upsampler, x.reshape(b, c, f * t), dtype).reshape(b, -1, f, t); xt = conv1d(module.channel_upsampler_t, xt, dtype)
        x, xt = _cross_transformer(module.crosstransformer, x, xt, dtype)
        if module.bottom_channels: b, c, f, t = x.shape; x = conv1d(module.channel_downsampler, x.reshape(b, c, f * t), dtype).reshape(b, -1, f, t); xt = conv1d(module.channel_downsampler_t, xt, dtype)
    for idx, decode in enumerate(module.decoder):
        skip, skip_length = saved.pop()
        x, pre = _hdec_layer(decode, x, skip, skip_length, dtype)
        offset = module.depth - len(module.tdecoder)
        if idx >= offset:
            tdec = module.tdecoder[idx - offset]
            length_t = lengths_t.pop()
            if tdec.empty: xt, _ = _hdec_layer(tdec, pre[:, :, 0], None, length_t, dtype)
            else: xt, _ = _hdec_layer(tdec, xt, saved_t.pop(), length_t, dtype)
    stems = len(module.sources)
    x = x.reshape(b, stems, -1, f_query, t)
    x = x * std[:, None] + mean[:, None]
    x = x.reshape(b, stems, -1, 2, f_query, t).transpose(0, 1, 2, 4, 5, 3)
    zout = x[..., 0] + (1j * x[..., 1])
    out_len = int(module.segment * module.samplerate) if module.use_train_segment else length
    x_audio = _demucs_ispec(module, zout, out_len, 0, dtype)
    xt = xt.reshape(b, stems, -1, out_len) * stdt[:, None] + meant[:, None]
    x_audio = xt + x_audio
    return x_audio[..., :length_pre_pad] if length_pre_pad else x_audio
def mlx_forward_demucs(module, raw_audio, dtype=torch.float16): return to_torch(mlx_forward_demucs_mx(module, to_mx(raw_audio, dtype=dtype), dtype), raw_audio)