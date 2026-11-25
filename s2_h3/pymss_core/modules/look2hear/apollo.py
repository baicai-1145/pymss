import torch
import torch.nn.functional as F
from torch import nn
from pymss_core.npu_fft import istft_npu, stft_npu, torch_stft_blocked
from ..mlx_backend import MpsBackendMixin
from pymss_core.runtime_cache import _clear_cache_dicts
def _cached_inference_tensor(module, name, tensor, input, version):
    # fp16/bf16 CUDA inference: memoize casted weights; keyed on _version so in-place param updates invalidate
    if tensor is None or (tensor.device == input.device and tensor.dtype == input.dtype): return tensor
    key = (name, input.device, input.dtype, version)
    cache = module.__dict__.setdefault("_pymss_apollo_inference_cache", {})
    cached = cache.get(name)
    if cached is not None and cached[0] == key: return cached[1]
    casted = tensor.detach().to(device=input.device, dtype=input.dtype)
    cache[name] = (key, casted)
    return casted
def _complex_from_ri(ri, dim): return torch.complex(ri.select(dim, 0).float(), ri.select(dim, 1).float())
def _complex_div_by_real(spec, denom): return spec / denom
def pointwise_conv1d(input, conv):
    # 1x1 conv1d -> linear: faster on CUDA fp16/bf16 inference
    if (conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups) != ((1,), (1,), (0,), (1,), 1): return conv(input)
    weight, bias = conv.weight[:, :, 0], conv.bias
    if input.is_cuda and input.dtype in (torch.float16, torch.bfloat16) and not torch.is_grad_enabled(): weight = _cached_inference_tensor(conv, 'pointwise_weight', weight, input, conv.weight._version); bias = _cached_inference_tensor(conv, 'pointwise_bias', bias, input, bias._version) if bias is not None else None
    return F.linear(input.transpose(1, 2), weight, bias).transpose(1, 2)
class RMSNorm(nn.Module):
    def __init__(self, dimension, groups=1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.groups, self.eps = groups, 1e-5
    def forward(self, input):
        B, N, T = input.shape
        assert N % self.groups == 0
        if self.groups == 1 and not torch.is_grad_enabled():
            x = input.transpose(1, 2)
            if input.is_cuda and input.dtype in (torch.float16, torch.bfloat16): weight = _cached_inference_tensor(self, 'rms_weight', self.weight, input, self.weight._version); return F.rms_norm(x, (N,), weight, self.eps).transpose(1, 2)
            return F.rms_norm(x, (N,), None, self.eps).transpose(1, 2).type_as(input) * self.weight.reshape(1, -1, 1)
        x = input.reshape(B, self.groups, -1, T).float()
        return (x * torch.rsqrt(x.square().mean(-2, keepdim=True) + self.eps)).type_as(input).reshape(B, N, T) * self.weight.reshape(1, -1, 1)
class RMVN(nn.Module):
    def __init__(self, dimension, groups=1):
        super().__init__()
        self.mean, self.std = nn.Parameter(torch.zeros(dimension)), nn.Parameter(torch.ones(dimension))
        self.groups, self.eps = groups, 1e-5
    def forward(self, input):
        B, N = input.shape[:2]
        assert N % self.groups == 0
        x = input.reshape(B, self.groups, N // self.groups, -1)
        norm = (x - x.mean(2, keepdim=True)) / (x.var(2, keepdim=True) + self.eps).sqrt()
        return norm.reshape(B, N, -1) * self.std.reshape(1, -1, 1) + self.mean.reshape(1, -1, 1)
class Roformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_head=8, theta=10000, window=10000, input_drop=0.0, attention_drop=0.0, causal=True):
        super().__init__()
        self.input_size, self.num_head, self.theta, self.window = input_size, num_head, theta, window
        self.hidden_size = hidden_size // num_head
        cos_freq, sin_freq = self._calc_rotary_emb()
        self.register_buffer("cos_freq", cos_freq)
        self.register_buffer("sin_freq", sin_freq)
        self.register_buffer("reverse_sign", torch.tensor([-1, 1]), persistent=False)
        self._rotary_freq_cache, self.attention_drop, self.causal, self.eps = {}, attention_drop, causal, 1e-5
        self.input_norm = RMSNorm(self.input_size)
        self.input_drop = nn.Dropout(p=input_drop)
        self.weight = nn.Conv1d(self.input_size, self.hidden_size * self.num_head * 3, 1, bias=False)
        self.output = nn.Conv1d(self.hidden_size * self.num_head, self.input_size, 1, bias=False)
        self.MLP = nn.Sequential(RMSNorm(self.input_size), nn.Conv1d(self.input_size, self.input_size * 8, 1, bias=False), nn.SiLU())
        self.MLP_output = nn.Conv1d(self.input_size * 4, self.input_size, 1, bias=False)
    def clear_runtime_cache(self): _clear_cache_dicts(self, "_rotary_freq_cache")
    def _calc_rotary_emb(self):
        freq = (1.0 / self.theta ** (torch.arange(0, self.hidden_size, 2)[: self.hidden_size // 2] / self.hidden_size)).reshape(1, -1)
        pos = torch.arange(self.window).reshape(-1, 1)
        return torch.cos(pos * freq).repeat_interleave(2, dim=-1), torch.sin(pos * freq).repeat_interleave(2, dim=-1)
    def _add_rotary_sequence(self, feature):
        T = feature.shape[-2]
        cos, sin = self._rotary_freq_cache.setdefault((T, feature.device, feature.dtype), (self.cos_freq[:T, 0::2].to(device=feature.device, dtype=feature.dtype).unsqueeze(0), self.sin_freq[:T, 0::2].to(device=feature.device, dtype=feature.dtype).unsqueeze(0)))
        if feature.dtype == torch.float32:
            rot = torch.complex(cos, sin)  # fused complex multiply: ~4.7x over strided even/odd writes
            return torch.view_as_real(torch.view_as_complex(feature.reshape(*feature.shape[:-1], -1, 2)) * rot).reshape_as(feature)
        output = torch.empty_like(feature)
        even, odd = feature[..., 0::2], feature[..., 1::2]
        torch.sub(even * cos, odd * sin, out=output[..., 0::2])
        torch.add(odd * cos, even * sin, out=output[..., 1::2])
        return output
    def forward(self, input):
        B, _, T = input.shape
        qkv = pointwise_conv1d(self.input_drop(self.input_norm(input)), self.weight)
        Q, K, V = torch.split(qkv.reshape(B, self.num_head, self.hidden_size * 3, T).mT, self.hidden_size, dim=-1)
        Q_rot, K_rot = self._add_rotary_sequence(Q), self._add_rotary_sequence(K)
        attention_output = F.scaled_dot_product_attention(Q_rot.contiguous(), K_rot.contiguous(), V.contiguous() if torch.is_grad_enabled() else V, dropout_p=self.attention_drop, is_causal=self.causal)
        output = pointwise_conv1d(attention_output.mT.reshape(B, -1, T), self.output) + input
        gate, z = self.MLP[2](pointwise_conv1d(self.MLP[0](output), self.MLP[1])).chunk(2, dim=1)
        return output + pointwise_conv1d(F.silu(gate) * z, self.MLP_output), (K_rot, V)
class ConvActNorm1d(nn.Module):
    def __init__(self, in_channel, hidden_channel, kernel=7, causal=False):
        super().__init__()
        self.in_channel, self.kernel, self.causal = in_channel, kernel, causal
        self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=kernel - 1 if causal else (kernel - 1) // 2, groups=in_channel), RMSNorm(in_channel), nn.Conv1d(in_channel, hidden_channel, 1), nn.SiLU(), nn.Conv1d(hidden_channel, in_channel, 1))
    def forward(self, input):
        y = pointwise_conv1d(self.conv[3](pointwise_conv1d(self.conv[1](self.conv[0](input)), self.conv[2])), self.conv[4])
        return input + y[..., : -self.kernel + 1] if self.causal else input + y
class ICB(nn.Module):
    def __init__(self, in_channel, kernel=7, causal=False):
        super().__init__()
        self.blocks = nn.Sequential(*[ConvActNorm1d(in_channel, in_channel * 4, kernel, causal=causal) for _ in range(3)])
    def forward(self, input): return self.blocks(input)
class BSNet(nn.Module):
    def __init__(self, feature_dim, kernel=7):
        super().__init__()
        self.feature_dim = feature_dim
        self.band_net = Roformer(self.feature_dim, self.feature_dim, num_head=8, window=100, causal=False)
        self.seq_net = ICB(self.feature_dim, kernel=kernel)
    def forward(self, input):
        B, nband, _, T = input.shape
        band, _ = self.band_net(input.permute(0, 3, 2, 1).reshape(B * T, -1, nband))
        band = band.reshape(B, T, -1, nband).permute(0, 3, 2, 1)
        return self.seq_net(band.reshape(B * nband, -1, T)).reshape(B, nband, -1, T)
class Apollo(MpsBackendMixin, nn.Module):
    def __init__(self, sr, win, feature_dim, layer):
        super().__init__()
        self.sr = sr
        self.win = int(sr * win // 1000)
        self.stride, self.enc_dim, self.feature_dim = self.win // 2, self.win // 2 + 1, feature_dim
        self.eps = torch.finfo(torch.float32).eps
        self.register_buffer("window", torch.hann_window(self.win), persistent=False)
        self._packed_cache = {}
        bandwidth = int(self.win / 160)
        self.band_width = [bandwidth] * 79 + [self.enc_dim - bandwidth * 79]
        self.nband = len(self.band_width)
        self.BN = nn.ModuleList([nn.Sequential(RMSNorm(width * 2 + 1), nn.Conv1d(width * 2 + 1, self.feature_dim, 1)) for width in self.band_width])
        self.net = nn.Sequential(*[BSNet(self.feature_dim) for _ in range(layer)])
        self.output = nn.ModuleList([nn.Sequential(RMSNorm(self.feature_dim), nn.Conv1d(self.feature_dim, width * 4, 1), nn.GLU(dim=1)) for width in self.band_width])
    def clear_runtime_cache(self): _clear_cache_dicts(self, "_packed_cache")
    def mlx_forward_mx(self, raw_audio):
        from ..apollo_mlx import mlx_forward_apollo_mx
        return mlx_forward_apollo_mx(self, raw_audio, self.mps_model_compute_dtype)
    def _window(self, input):
        if input.device.type == "npu":
            # torch_npu's hann_window hits an int64/float cast bug on device.
            return self.window.to(dtype=torch.float32, device=input.device)
        return self.window.to(device=input.device)
    def _uniform_band_prefix(self):
        width = self.band_width[0]
        return next((i for i, w in enumerate(self.band_width) if w != width), self.nband), width
    def _use_packed_band_ops(self): return not self.training and not torch.is_grad_enabled() and self._uniform_band_prefix()[0] > 1
    def _stft(self, input):
        B, nch, nsample = input.shape
        fft_fallback = torch_stft_blocked(input.device)
        fn = stft_npu if fft_fallback else torch.stft
        flat = input.view(B * nch, nsample)
        if fft_fallback: flat = flat.float()
        return fn(flat, n_fft=self.win, hop_length=self.stride, window=self._window(input), return_complex=True)
    def _band_norm_power(self, spec, band_idx, width):
        this_spec = spec[:, band_idx : band_idx + width]
        power = ((this_spec.real.square() + this_spec.imag.square()).sum(1, keepdim=True) + self.eps).sqrt()  # B,1,T
        return _complex_div_by_real(this_spec, power), power
    def _cached_packed_modules(self, name, modules, count):
        conv = modules[0][1]
        key, cached = (name, count, conv.weight.device, conv.weight.dtype), self._packed_cache.get(name)
        if cached is not None and cached["key"] == key: return cached["norm_weight"], cached["conv_weight"], cached["conv_bias"], cached["groups"], cached["eps"]
        modules = list(modules[:count])
        packed = { "key": key, "norm_weight": torch.stack([module[0].weight.detach() for module in modules]), "conv_weight": torch.cat([module[1].weight.detach() for module in modules]), "conv_bias": torch.cat([module[1].bias.detach() for module in modules]) if modules[0][1].bias is not None else None, "groups": modules[0][0].groups, "eps": modules[0][0].eps, }
        self._packed_cache[name] = packed
        return packed["norm_weight"], packed["conv_weight"], packed["conv_bias"], packed["groups"], packed["eps"]
    @staticmethod
    def _packed_rms_norm(input, weight, groups, eps):
        b, bands, c, frames = input.shape
        x = input.reshape(b, bands, groups, c // groups, frames).float()
        return (x * torch.rsqrt(x.square().mean(3, keepdim=True) + eps)).to(dtype=input.dtype).reshape(b, bands, c, frames) * weight.reshape(1, bands, c, 1)
    def _packed_bn_prefix(self, input, count):
        b, bands, c, frames = input.shape
        norm_weight, conv_weight, conv_bias, groups, eps = self._cached_packed_modules("bn", self.BN, count)
        return F.conv1d(self._packed_rms_norm(input, norm_weight, groups, eps).reshape(b, bands * c, frames), conv_weight, conv_bias, groups=bands).reshape(b, bands, self.feature_dim, frames)
    def _packed_output_prefix(self, feature, count, width):
        b, bands, c, frames = feature.shape
        norm_weight, conv_weight, conv_bias, groups, eps = self._cached_packed_modules("output", self.output, count)
        output = F.conv1d(self._packed_rms_norm(feature, norm_weight, groups, eps).reshape(b, bands * c, frames), conv_weight, conv_bias, groups=bands).reshape(b, bands, width * 4, frames)
        return F.glu(output, dim=2).reshape(b, bands, 2, width, frames)
    def spec_band_split(self, input):
        spec, norms, powers, band_idx = self._stft(input), [], [], 0
        for width in self.band_width:
            norm, power = self._band_norm_power(spec, band_idx, width)
            norms.append(norm); powers.append(power); band_idx += width
        return norms, torch.cat(powers, 1)
    def _spec_band_split_packed(self, input):
        spec = self._stft(input)
        count, width = self._uniform_band_prefix()
        prefix_spec = spec[:, : count * width].reshape(spec.shape[0], count, width, -1)
        # BIREN has no complex abs(); the power is the same on the real planes.
        prefix_power = (torch.view_as_real(prefix_spec).pow(2).sum(-1).sum(2) + self.eps).sqrt()
        prefix_norm = _complex_div_by_real(prefix_spec, prefix_power.unsqueeze(2))
        tail_norm, tail_power, band_idx = [], [], count * width
        for width in self.band_width[count:]:
            norm, power = self._band_norm_power(spec, band_idx, width)
            tail_norm.append(norm)
            tail_power.append(power)
            band_idx += width
        return prefix_norm, prefix_power, tail_norm, tail_power
    def feature_extractor(self, input): return self._feature_extractor_packed(input) if self._use_packed_band_ops() else self._feature_extractor_by_band(input)
    def _feature_extractor_by_band(self, input):
        subband_norm, subband_power = self.spec_band_split(input)
        return torch.stack([ self.BN[i](torch.cat([subband_norm[i].real, subband_norm[i].imag, torch.log(subband_power[:, i].unsqueeze(1))], 1)) for i in range(self.nband) ], 1)
    def _feature_extractor_packed(self, input):
        prefix_norm, prefix_power, tail_norm, tail_power = self._spec_band_split_packed(input)
        count, _ = self._uniform_band_prefix()
        prefix = self._packed_bn_prefix(torch.cat([prefix_norm.real, prefix_norm.imag, torch.log(prefix_power).unsqueeze(2)], dim=2), count)
        if count == self.nband: return prefix
        return torch.cat([ prefix, torch.stack([ self.BN[count + offset](torch.cat([norm.real, norm.imag, torch.log(tail_power[offset])], 1)) for offset, norm in enumerate(tail_norm) ], 1), ], dim=1)
    def _estimate_spec_by_band(self, feature, batch_channels): return torch.cat([ _complex_from_ri(output(feature[:, i]).view(batch_channels, 2, width, -1), dim=1) for i, (output, width) in enumerate(zip(self.output, self.band_width)) ], 1)
    def _estimate_spec_packed(self, feature, batch_channels):
        count, width = self._uniform_band_prefix()
        prefix = _complex_from_ri(self._packed_output_prefix(feature[:, :count], count, width), dim=2).reshape(batch_channels, count * width, -1)
        if count == self.nband: return prefix
        return torch.cat([prefix] + [ _complex_from_ri(self.output[i](feature[:, i]).view(batch_channels, 2, self.band_width[i], -1), dim=1) for i in range(count, self.nband) ], 1)
    def forward(self, input):
        if self._use_mlx_full_forward(input):
            try:
                from ..apollo_mlx import mlx_forward_apollo
                return mlx_forward_apollo(self, input, self.mps_model_compute_dtype)
            except Exception as exc:
                self._pymss_mlx_full_backend_error = repr(exc)
                self.mps_model_backend = "torch"
        B, nch, nsample = input.shape
        feature = self.net(self.feature_extractor(input))
        est_spec = self._estimate_spec_packed(feature, B * nch) if self._use_packed_band_ops() else self._estimate_spec_by_band(feature, B * nch)
        istft_fn = istft_npu if torch_stft_blocked(input.device) else torch.istft
        return istft_fn(est_spec.to(dtype=torch.complex64), n_fft=self.win, hop_length=self.stride, window=self._window(input), length=nsample).view(B, nch, -1)
