import torch
from ..mlx_backend import (_cache_limit, conv1d, conv2d, glu, instance_norm2d, linear, maybe_eval, mx_dtype, overlap_add, param, periodic_hann_window, reflect_pad_last, silu, to_mx, to_torch)
from . import hyperace_segm
from .bands import contiguous_dim_groups
from .bs_roformer_hyperace import BSRoformerHyperACE
from .conformer import Conformer
from .mel_band_roformer import MelBandRoformer
from .mlx_attention import _COMPUTE_DTYPE, _mlx_attention, _mlx_feed_forward, _mlx_output_norm, _rms_norm
torch_to_mlx_input = to_mx
def _cache_key(params, dtype): return tuple(None if p is None else (p.data_ptr(), p._version, tuple(p.shape), dtype) for p in params)
def _padded_window(win_length, n_fft, dtype):
    import mlx.core as mx
    window = periodic_hann_window(win_length, dtype)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        window = mx.pad(window, [(left, n_fft - win_length - left)])
    elif win_length > n_fft: raise ValueError("MLX RoFormer STFT does not support win_length > n_fft")
    return window
def _stft_roformer(module, raw_audio, dtype):
    import mlx.core as mx
    import numpy as np
    if raw_audio.ndim == 2: raw_audio = raw_audio[:, None, :]
    batch, channels, audio_length = raw_audio.shape
    if (module.stereo and channels != 2) or (not module.stereo and channels != 1): raise ValueError("raw_audio channel count does not match RoFormer stereo setting")
    kw = module.stft_kwargs
    n_fft, hop, win_length = int(kw["n_fft"]), int(kw["hop_length"]), int(kw["win_length"])
    normalized = bool(kw.get("normalized", False))
    flat = reflect_pad_last(raw_audio.reshape(batch * channels, audio_length).astype(dtype), n_fft // 2, n_fft // 2)
    frames = 1 + (flat.shape[-1] - n_fft) // hop
    framed = mx.as_strided(flat, shape=(flat.shape[0], frames, n_fft), strides=(flat.shape[-1], hop, 1))
    window = _padded_window(win_length, n_fft, dtype)
    stft = mx.fft.rfft(framed * window, n=n_fft, axis=-1)
    if normalized: stft = stft / np.sqrt(n_fft)
    stft = mx.moveaxis(stft, -1, -2)  # (n, T, F) -> (n, F, T)
    ri = mx.stack((stft.real, stft.imag), axis=-1)
    freq_bins = ri.shape[-3]
    ri = mx.transpose(ri.reshape(batch, channels, freq_bins, frames, 2), (0, 2, 1, 3, 4))
    return ri.reshape(batch, freq_bins * channels, frames, 2), { "batch": batch, "channels": channels, "freq_bins": freq_bins, "audio_length": audio_length, "window": window, "n_fft": n_fft, "hop": hop, "normalized": normalized, "dtype": dtype, }
def _istft_roformer(module, stft_repr, context, length):
    import mlx.core as mx
    b, n, _, t, _ = stft_repr.shape
    channels, freq_bins, n_fft, hop, dtype = context["channels"], context["freq_bins"], context["n_fft"], context["hop"], context["dtype"]
    ri = mx.transpose(stft_repr.reshape(b, n, freq_bins, channels, t, 2), (0, 1, 3, 2, 4, 5)).reshape(b * n * channels, freq_bins, t, 2)
    complex_stft = ri[..., 0] + (1j * ri[..., 1])
    if getattr(module, "zero_dc", False): complex_stft = complex_stft.at[:, 0, :].multiply(0)
    if context["normalized"]: complex_stft = complex_stft * context['n_fft'] ** 0.5
    frames = mx.fft.irfft(mx.moveaxis(complex_stft, -2, -1), n=n_fft, axis=-1).astype(dtype) * context["window"]
    mx.eval(frames)
    del complex_stft, ri
    audio = overlap_add(frames, context["window"], hop)
    mx.eval(audio)
    del frames
    pad = n_fft // 2
    audio = audio[..., pad:-pad] if length is None and pad > 0 else audio[..., pad : pad + length]
    audio = audio.reshape(context["batch"], n, channels, audio.shape[-1])
    return audio[:, 0] if n == 1 else audio
def _band_split_cache(module, dtype):
    cache = getattr(module, "_pymss_mlx_full_band_split_cache", None)
    params = [p for norm, linear in module.band_split.to_features for p in (norm.gamma, linear.weight, linear.bias)]
    key = (tuple(module.band_split.dim_inputs), _cache_key(params, dtype))
    if cache is not None and cache.get("key") == key: return cache
    groups = []
    for start, end, dim_in in contiguous_dim_groups(module.band_split.dim_inputs):
        norms = [module.band_split.to_features[i][0] for i in range(start, end)]
        linears = [module.band_split.to_features[i][1] for i in range(start, end)]
        groups.append({ "start": start, "end": end, "dim_in": dim_in, "offset_start": module.band_split._dim_offsets[start], "offset_end": module.band_split._dim_offsets[end], "gamma": to_mx(torch.stack([n.gamma for n in norms]), dtype), "weight": to_mx(torch.stack([lin.weight for lin in linears]), dtype), "bias": None if linears[0].bias is None else to_mx(torch.stack([lin.bias for lin in linears]), dtype), })
    cache = {"key": key, "groups": groups}
    module._pymss_mlx_full_band_split_cache = cache
    return cache
def _grouped_linear(x, weight, bias): import mlx.core as mx; out = mx.einsum("...gi,goi->...go", x, weight); return out if bias is None else out + bias
def _band_split(module, x, dtype):
    import mlx.core as mx
    outs = []
    for group in _band_split_cache(module, dtype)["groups"]: group_x = x[..., group["offset_start"] : group["offset_end"]]; group_x = group_x.reshape(*group_x.shape[:-1], group["end"] - group["start"], group["dim_in"]); outs.append(_grouped_linear(_rms_norm(group_x, group["gamma"]), group["weight"], group["bias"]))
    # Keep dense skip residuals wide from their first projection: narrowing
    # here could overflow before _forward_mask_core can restore FP32.
    out = mx.concatenate(outs, axis=-2)
    if getattr(module, "skip_connection", False): return out
    # Other models restore the requested precision before the transformer.
    return out.astype(mx_dtype(dtype))
def _transformer(module, x, dtype):
    for attn, ff in module.layers:
        x = _mlx_attention(attn, x, dtype) + x
        x = _mlx_feed_forward(ff, x, dtype) + x
        maybe_eval(x)
    return _mlx_output_norm(module.norm, x, dtype)
def _conformer(module, x, dtype):
    for block in module.layers:
        x = x + _macaron_ff(block.ff1, x, dtype)
        x = x + _mlx_attention(block.attn, x, dtype)
        x = x + _conformer_conv(block.conv, x, dtype)
        x = x + _macaron_ff(block.ff2, x, dtype)
        x = _mlx_output_norm(block.out_norm, x, dtype)
        maybe_eval(x)
    return _mlx_output_norm(module.norm, x, dtype)
def _macaron_ff(module, x, dtype): return _mlx_feed_forward(module.ff, x, dtype) * module.scale
def _conformer_conv(module, x, dtype):
    # Sequential indices match ConformerConvModule / MSST checkpoints.
    norm, _, pointwise_in, _, depthwise, batch_norm, _, pointwise_out, _, _ = module.net
    y = _rms_norm(x, param(norm, "gamma", norm.gamma, dtype))
    if y.dtype != mx_dtype(dtype): y = y.astype(mx_dtype(dtype))
    y = y.transpose(0, 2, 1)
    y = conv1d(pointwise_in, y, dtype)
    y = glu(y, axis=1)
    y = batch_norm1d(batch_norm, conv1d(depthwise, y, dtype), dtype)
    y = conv1d(pointwise_out, silu(y), dtype)
    return y.transpose(0, 2, 1)
def batch_norm1d(module, x, dtype):
    import mlx.core as mx
    if module.training: raise TypeError("MLX Conformer BatchNorm1d supports eval mode only")
    y = x.astype(mx.float32)
    mean = param(module, 'running_mean', module.running_mean, torch.float32).reshape(1, -1, 1)
    var = param(module, 'running_var', module.running_var, torch.float32).reshape(1, -1, 1)
    y = (y - mean) * mx.rsqrt(var + module.eps)
    if module.affine: y = y.astype(x.dtype) * param(module, 'weight', module.weight, dtype).reshape(1, -1, 1); y = y + param(module, 'bias', module.bias, dtype).reshape(1, -1, 1)
    return y.astype(x.dtype)
def _sequence_model(module, x, dtype): return _conformer(module, x, dtype) if isinstance(module, Conformer) else _transformer(module, x, dtype)
def _final_norm(module, x, dtype): return x if (isinstance(module.final_norm, torch.nn.Identity)) else _rms_norm(x, param(module.final_norm, "gamma", module.final_norm.gamma, dtype))
def _mask_estimator_layers(mlp_with_glu):
    layers = []
    mlp, glu_mod = mlp_with_glu
    if not isinstance(glu_mod, torch.nn.GLU): raise TypeError("MLX RoFormer mask estimator expects nn.GLU")
    for layer in mlp:
        if isinstance(layer, torch.nn.Linear):
            layers.append(("linear", layer))
        elif isinstance(layer, torch.nn.Tanh): layers.append(('tanh', None))
        else: raise TypeError(f'unsupported MLX RoFormer mask estimator layer: {type(layer).__name__}')
    return tuple(layers)
def _mask_estimator_cache(estimator, dtype):
    cache = getattr(estimator, "_pymss_mlx_full_mask_cache", None)
    params = [ p for mlp_with_glu in estimator.to_freqs for kind, layer in _mask_estimator_layers(mlp_with_glu) if kind == "linear" for p in (layer.weight, layer.bias) ]
    key = (tuple(estimator.dim_inputs), _cache_key(params, dtype))
    if cache is not None and cache.get("key") == key: return cache
    # per-band loop: at real mel banding the width-groups fragment (39 groups / 48 bands), so stacked
    # einsum is performance-neutral vs plain per-band matmuls (tie at controlled load; <=2% worse under
    # sustained thermal load). Kept per-band: simpler and never slower.
    band_layers = []
    for mlp_with_glu in estimator.to_freqs:
        layers = []
        for kind, layer in _mask_estimator_layers(mlp_with_glu):
            if kind == "tanh": layers.append(('tanh', None, None))
            else: layers.append(('linear', to_mx(layer.weight, dtype), None if layer.bias is None else to_mx(layer.bias, dtype)))
        band_layers.append(tuple(layers))
    cache = {"key": key, "band_layers": tuple(band_layers)}
    estimator._pymss_mlx_full_mask_cache = cache
    return cache
def segm_eval(*arrays):  # eval points inside the segm tree; no-op without mlx (mocked test paths)
    try:
        import mlx.core as mx
    except Exception:
        return
    mx.eval(*arrays)
def _mask_estimator(estimator, x, dtype):
    import mlx.core as mx
    outs = []
    for band_index, layers in enumerate(_mask_estimator_cache(estimator, dtype)["band_layers"]):
        group_x = x[:, :, band_index, :]
        for kind, weight, bias in layers: group_x = mx.tanh(group_x) if kind == "tanh" else linear(group_x, weight, bias)
        out = glu(group_x, axis=-1)
        segm_eval(out)
        outs.append(out)
    out = mx.concatenate(outs, axis=-1)
    segm_eval(out)
    return out
def _conv_block(module, x, dtype): x = conv2d(module.conv, x, dtype); x = instance_norm2d(module.bn, x, dtype); return x if isinstance(module.act, torch.nn.Identity) else silu(x)
def _dsconv_block(module, x, dtype): x = conv2d(module.pwconv, conv2d(module.dwconv, x, dtype), dtype); x = instance_norm2d(module.bn, x, dtype); return x if isinstance(module.act, torch.nn.Identity) else silu(x)
def _resize_positions(in_size, out_size): import mlx.core as mx; pos = (mx.arange(out_size, dtype=mx.float32) + 0.5) * (in_size / out_size) - 0.5; lower = mx.floor(pos); weight = pos - lower; return (mx.clip(lower, 0, in_size - 1).astype(mx.int32), mx.clip(lower + 1, 0, in_size - 1).astype(mx.int32), weight)
def _resize_bilinear_nchw(x, size):
    import mlx.core as mx
    out_h, out_w = int(size[0]), int(size[1])
    in_h, in_w = x.shape[2], x.shape[3]
    if in_h == out_h and in_w == out_w: return x
    y0, y1, wy = _resize_positions(in_h, out_h)
    x0, x1, wx = _resize_positions(in_w, out_w)
    def corner(yy, xx): return mx.take(mx.take(x, yy, axis=2), xx, axis=3)
    wy, wx = wy.reshape(1, 1, out_h, 1), wx.reshape(1, 1, 1, out_w)
    r0 = corner(y0, x0) * (1 - wx) + corner(y0, x1) * wx
    r1 = corner(y1, x0) * (1 - wx) + corner(y1, x1) * wx
    return r0 * (1 - wy) + r1 * wy
def _seq(module, x, dtype):
    for child in module: x = _segm_module(child, x, dtype)
    return x
def _ds_bottleneck(module, x, dtype): y = _dsconv_block(module.dsconv2, _dsconv_block(module.dsconv1, x, dtype), dtype); return x + y if module.shortcut else y
def _ds_c3k(module, x, dtype): import mlx.core as mx; return _conv_block(module.cv3, mx.concatenate((_seq(module.m, _conv_block(module.cv1, x, dtype), dtype), _conv_block(module.cv2, x, dtype)), axis=1), dtype)
def _ds_c3k2(module, x, dtype): return _conv_block(module.cv2, _ds_c3k(module.m, _conv_block(module.cv1, x, dtype), dtype), dtype)
def _adaptive_hyperedge_generation(module, x, dtype):
    import mlx.core as mx
    b, n, c = x.shape
    context = mx.concatenate((mx.mean(x, axis=1), mx.max(x, axis=1)), axis=1)
    proto = param(module, "global_proto", module.global_proto, dtype)[None] + linear(context, param(module.context_mapper, "weight", module.context_mapper.weight, dtype)).reshape(b, module.num_hyperedges, c)
    z = linear(x, param(module.query_proj, "weight", module.query_proj.weight, dtype))
    z = z.reshape(b, n, module.num_heads, module.head_dim).transpose(0, 2, 1, 3)
    proto = proto.reshape(b, module.num_hyperedges, module.num_heads, module.head_dim).transpose(0, 2, 3, 1)
    return mx.softmax(mx.mean((z @ proto) * module.scale, axis=1).transpose(0, 2, 1), axis=-1)
def _hypergraph_convolution(module, x, a, dtype):
    hidden = a @ x
    hidden = silu(linear(hidden, param(module.W_e, "weight", module.W_e.weight, dtype)))
    hidden = linear(a.transpose(0, 2, 1) @ hidden, param(module.W_v, "weight", module.W_v.weight, dtype))
    return x + silu(hidden)
def _adaptive_hypergraph_computation(module, x, dtype): b, _, h, w = x.shape; x_flat = x.reshape(b, x.shape[1], h * w).transpose(0, 2, 1); a = _adaptive_hyperedge_generation(module.adaptive_hyperedge_gen, x_flat, dtype); return _hypergraph_convolution(module.hypergraph_conv, x_flat, a, dtype).transpose(0, 2, 1).reshape(b, -1, h, w)
def _c3ah(module, x, dtype): import mlx.core as mx; return _conv_block(module.cv3, mx.concatenate((_adaptive_hypergraph_computation(module.ahc, _conv_block(module.cv2, x, dtype), dtype), _conv_block(module.cv1, x, dtype)), axis=1), dtype)
def _hyperace(module, features, dtype):
    import mlx.core as mx
    b2, b3, b4, b5 = features
    size = b4.shape[2:]
    x = _conv_block(module.fuse_conv, mx.concatenate((_resize_bilinear_nchw(b2, size), _resize_bilinear_nchw(b3, size), b4, _resize_bilinear_nchw(b5, size)), axis=1), dtype); segm_eval(x)
    x_h, x_l, x_s = x[:, : module.c_h], x[:, module.c_h : module.c_h + module.c_l], x[:, module.c_h + module.c_l :]
    high = _conv_block(module.high_order_fuse, mx.concatenate([_c3ah(branch, x_h, dtype) for branch in module.high_order_branch], axis=1), dtype); segm_eval(high)
    out = _conv_block(module.final_fuse, mx.concatenate((high, _seq(module.low_order_branch, x_l, dtype), x_s), axis=1), dtype); segm_eval(out)
    return out
def _gated_fusion(module, f_in, h, dtype): return f_in + param(module, "gamma", module.gamma, dtype) * h
def _backbone(module, x, dtype):
    x2 = _seq(module.p2, _dsconv_block(module.stem, x, dtype), dtype); segm_eval(x2)
    x3 = _seq(module.p3, x2, dtype); segm_eval(x3)
    x4 = _seq(module.p4, x3, dtype); segm_eval(x4)
    x5 = _seq(module.p5, x4, dtype); segm_eval(x5)
    return [x2, x3, x4, x5]
def _decoder(module, enc_feats, h_ace, dtype):
    p2, p3, p4, p5 = enc_feats
    d5 = _gated_fusion(module.fusion_d5, _conv_block(module.skip_p5, p5, dtype), _conv_block(module.h_to_d5, _resize_bilinear_nchw(h_ace, p5.shape[2:]), dtype), dtype); segm_eval(d5)
    d4 = _ds_c3k2(module.up_d5, _resize_bilinear_nchw(d5, p4.shape[2:]), dtype) + _conv_block(module.skip_p4, p4, dtype)
    d4 = _gated_fusion(module.fusion_d4, d4, _conv_block(module.h_to_d4, _resize_bilinear_nchw(h_ace, d4.shape[2:]), dtype), dtype); segm_eval(d4)
    d3 = _ds_c3k2(module.up_d4, _resize_bilinear_nchw(d4, p3.shape[2:]), dtype) + _conv_block(module.skip_p3, p3, dtype)
    d3 = _gated_fusion(module.fusion_d3, d3, _conv_block(module.h_to_d3, _resize_bilinear_nchw(h_ace, d3.shape[2:]), dtype), dtype); segm_eval(d3)
    d2 = _ds_c3k2(module.up_d3, _resize_bilinear_nchw(d3, p2.shape[2:]), dtype) + _conv_block(module.skip_p2, p2, dtype)
    d2 = _gated_fusion(module.fusion_d2, d2, _conv_block(module.h_to_d2, _resize_bilinear_nchw(h_ace, d2.shape[2:]), dtype), dtype); segm_eval(d2)
    out = _ds_c3k2(module.final_d2, d2, dtype); segm_eval(out)
    return out
def _tfc_tdf(module, x, dtype):
    for block in module.blocks: shortcut = conv2d(block.shortcut, x, dtype); x = _segm_module(block.tfc1, x, dtype); x = _segm_module(block.tfc2, x + _segm_module(block.tdf, x, dtype), dtype) + shortcut
    return x
def _freq_pixel_shuffle(module, x, dtype): x = _dsconv_block(module.conv, x, dtype); b, c_r, h, w = x.shape; out_c = c_r // module.scale; x = x.reshape(b, out_c, module.scale, h, w).transpose(0, 1, 3, 4, 2).reshape(b, out_c, h, w * module.scale); return _tfc_tdf(module.out_conv, x, dtype)
def _progressive_upsample_head(module, x, dtype):
    x = _freq_pixel_shuffle(module.block1, x, dtype); segm_eval(x)
    x = _freq_pixel_shuffle(module.block2, x, dtype); segm_eval(x)
    x = _freq_pixel_shuffle(module.block3, x, dtype); segm_eval(x)
    x = _freq_pixel_shuffle(module.block4, x, dtype); segm_eval(x)
    if x.shape[-1] != module.target_bins: x = _resize_bilinear_nchw(x, (x.shape[2], module.target_bins))
    out = conv2d(module.final_conv, x, dtype); segm_eval(out)
    return out
def _segm_model(module, x, dtype): enc_feats = _backbone(module.backbone, x, dtype); dec_feat = _decoder(module.decoder, enc_feats, _hyperace(module.hyperace, enc_feats, dtype), dtype); dec_feat = _resize_bilinear_nchw(dec_feat, (x.shape[2], dec_feat.shape[-1])); return _progressive_upsample_head(module.upsample_head, dec_feat, dtype)
def _segm_module(module, x, dtype):
    handlers = {torch.nn.Sequential: _seq, hyperace_segm.Conv: _conv_block, hyperace_segm.DSConv: _dsconv_block, hyperace_segm.DS_Bottleneck: _ds_bottleneck, hyperace_segm.DS_C3k: _ds_c3k, hyperace_segm.DS_C3k2: _ds_c3k2, hyperace_segm.TFC_TDF: _tfc_tdf, torch.nn.InstanceNorm2d: instance_norm2d}
    for klass, fn in handlers.items():
        if isinstance(module, klass): return fn(module, x, dtype)
    if isinstance(module, torch.nn.SiLU): return silu(x)
    if isinstance(module, torch.nn.Conv2d): return conv2d(module, x, dtype)
    if isinstance(module, torch.nn.Linear): return linear(x, param(module, "weight", module.weight, dtype), None if module.bias is None else param(module, "bias", module.bias, dtype))
    if isinstance(module, torch.nn.Identity): return x
    raise TypeError(f"unsupported HyperACE SegmModel layer for MLX full backend: {type(module).__name__}")
def _estimate_masks(module, x, dtype):
    import mlx.core as mx
    if isinstance(module, BSRoformerHyperACE) and module.mask_mode != "no_segm":
        masks = []
        segm_input = mx.array(x.transpose(0, 3, 1, 2))  # contiguous NCHW: avoids per-level re-striding of the view
        for estimator in module.mask_estimators:
            segm = _segm_model(estimator.segm, segm_input, dtype)
            segm = segm.transpose(0, 2, 3, 1).reshape(segm.shape[0], segm.shape[2], -1)
            maybe_eval(segm)
            masks.append(segm if module.mask_mode == "segm_only" else _mask_estimator(estimator, x, dtype) + segm)
        return mx.stack(masks, axis=1)
    outs = []
    for estimator in module.mask_estimators:
        out = _mask_estimator(estimator, x, dtype)
        maybe_eval(out)
        outs.append(out)
    return mx.stack(outs, axis=1)
def _mask_to_complex_shape(mask): return mask.reshape(*mask.shape[:3], mask.shape[3] // 2, 2).transpose(0, 1, 3, 2, 4)
def _forward_mask_core(module, stft_repr, dtype):
    b, fs, model_t, complex_dim = stft_repr.shape
    x = _band_split(module, stft_repr.transpose(0, 2, 1, 3).reshape(b, model_t, fs * complex_dim), dtype)
    residual_store = [] if getattr(module, "skip_connection", False) else None
    if residual_store is not None and x.dtype != stft_repr.dtype:
        # Dense cross-layer skips can exceed FP16 range. Keep residuals at FFT
        # precision; attention/FFN normalize in that precision, then cast their
        # bounded activations before GEMM instead of downcasting the residual.
        x = x.astype(stft_repr.dtype)
    for time_transformer, freq_transformer in module.layers:
        if residual_store is not None:
            for residual in residual_store: x = x + residual
        b, t, f, d = x.shape
        x = _sequence_model(time_transformer, x.transpose(0, 2, 1, 3).reshape(b * f, t, d), dtype)
        x = x.reshape(b, f, t, d).transpose(0, 2, 1, 3)
        x = _sequence_model(freq_transformer, x.reshape(b * t, f, d), dtype).reshape(b, t, f, d)
        maybe_eval(x)
        if residual_store is not None: residual_store.append(x)
    x_norm = _final_norm(module, x, dtype)
    del x
    return _mask_to_complex_shape(_estimate_masks(module, x_norm, dtype))
def _complex_from_ri(x): return x[..., 0] + (1j * x[..., 1])
def _ri_from_complex(x): import mlx.core as mx; return mx.stack((x.real, x.imag), axis=-1)
def _mask_stft_repr_bsr(module, stft_repr, dtype): mask = _forward_mask_core(module, stft_repr, dtype); return _complex_from_ri(stft_repr[:, None]) * _complex_from_ri(mask)
def _mbr_buffer_cache(module, dtype):
    cache = getattr(module, "_pymss_mlx_full_mbr_cache", None)
    key = (module.freq_indices.data_ptr(), module.freq_indices._version, module.num_bands_per_channel_freq.data_ptr(), module.num_bands_per_channel_freq._version, str(dtype))
    if cache is not None and cache.get("key") == key: return cache
    cache = { "key": key, "freq_indices": to_mx(module.freq_indices, torch.int32), "denom": to_mx(module.num_bands_per_channel_freq, dtype)[..., None] }
    module._pymss_mlx_full_mbr_cache = cache
    return cache
def _mask_stft_repr_mbr(module, stft_repr, context, dtype):
    import mlx.core as mx
    bufs = _mbr_buffer_cache(module, dtype)
    freq_indices, denom = bufs["freq_indices"], bufs["denom"]
    # Overlapping band masks are spectral DSP; accumulate at FFT precision even
    # when the neural layers use FP16.
    masks = _forward_mask_core(module, stft_repr[:, freq_indices], dtype)
    if masks.dtype != stft_repr.dtype:
        masks = masks.astype(stft_repr.dtype)
        mx.eval(masks)  # Drop the low-precision mask graph before DSP accumulation.
    masks_summed = mx.zeros((context["batch"], len(module.mask_estimators), stft_repr.shape[1], stft_repr.shape[-2], 2), dtype=masks.dtype).at[:, :, freq_indices, :, :].add(masks)
    return _complex_from_ri(stft_repr[:, None]) * _complex_from_ri(masks_summed / mx.maximum(denom, 1e-8))
def mlx_forward_roformer_mx(module, raw_audio, dtype=_COMPUTE_DTYPE):
    """Run MLX inference at the requested neural precision, with wider DSP/skips.

    FP16 is not a guarantee of FP32-equivalent outputs. In particular, the
    HyperACE ``full``/``segm_only`` segmentation heads can amplify quantization
    differences; use explicit FP32 for fidelity checks on those branches.
    """
    if getattr(module, "use_pope", False): raise NotImplementedError("PoPE models require the PyTorch backend")
    if dtype not in (torch.float16, torch.float32): raise TypeError("MLX full RoFormer supports torch.float16 or torch.float32 compute dtype")
    import mlx.core as mx
    _cache_limit()  # Bound allocator reuse before even the first warmup/STFT stage.
    mx_dtype = mx.float16 if dtype == torch.float16 else mx.float32
    stft_repr, context = _stft_roformer(module, raw_audio.astype(mx_dtype), mx_dtype)
    mx.eval(stft_repr)
    if isinstance(module, MelBandRoformer):
        masked = _mask_stft_repr_mbr(module, stft_repr, context, dtype)
        length = context["audio_length"] if module.match_input_audio_length else None
    else:
        masked = _mask_stft_repr_bsr(module, stft_repr, dtype)
        length = context["audio_length"]
    masked = _ri_from_complex(masked)
    mx.eval(masked)
    del stft_repr
    return _istft_roformer(module, masked, context, length)
def mlx_forward_roformer(module, raw_audio, dtype=_COMPUTE_DTYPE): return to_torch(mlx_forward_roformer_mx(module, to_mx(raw_audio, dtype=dtype), dtype), raw_audio)
