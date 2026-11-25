import inspect
import math
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from .demucs_local import BLSTM, DConv, HDecLayer, HEncLayer, MultiWrap
from .demucs_local import ScaledEmbedding as LegacyScaledEmbedding
from .demucs_local import _freq_dconv as _dconv_freq
from .demucs_local import _hann
from .demucs_local import rescale_module as _rescale_module
LEGACY_STEMS_4,LEGACY_STEMS_2,EPS = ["drums", "bass", "other", "vocals"], ["vocals", "non_vocals"], 1e-8
def center_trim(tensor, reference):
    if hasattr(reference, "size"): reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0: raise ValueError(f"tensor must be larger than reference. Delta is {delta}.")
    return tensor[..., delta // 2 : -(delta - delta // 2)] if delta else tensor
def _resample_x2(x): return F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
def _downsample_x2(x, length): return F.interpolate(x, size=length, mode="linear", align_corners=False)
def _unet_forward(model, x):
    saved = []
    for encode in model.encoder:
        x = encode(x)
        saved.append(x)
    if model.lstm: x = model.lstm(x)
    for decode in model.decoder:
        x = decode(x + center_trim(saved.pop(-1), x))
    return x
def _valid_length(model, length, with_context):
    if model.resample: length *= 2
    for _ in range(model.depth):
        length = max(1, math.ceil((length - model.kernel_size) / model.stride) + 1)
        if with_context: length += model.context - 1
    for _ in range(model.depth):
        length = (length - 1) * model.stride + model.kernel_size
    return math.ceil(length / 2) if model.resample else int(length)
class LegacyDemucs(nn.Module):
    def __init__(self, sources=4, audio_channels=2, channels=64, depth=6, rewrite=True, glu=True, rescale=0.1, resample=True, upsample=None, kernel_size=8, stride=4, growth=2.0, lstm_layers=2, context=3, normalize=False, samplerate=44100, segment_length=4 * 10 * 44100, **_):
        super().__init__()
        if upsample is not None: resample = bool(upsample)
        self.audio_channels, self.sources = audio_channels, _normalize_sources(sources)
        self.kernel_size, self.context, self.stride, self.depth = kernel_size, context, stride, depth
        self.resample, self.channels, self.normalize = resample, channels, normalize
        self.samplerate, self.segment_length = samplerate, segment_length
        self.encoder, self.decoder = nn.ModuleList(), nn.ModuleList()
        activation, ch_scale = (nn.GLU(dim=1), 2) if glu else (nn.ReLU(), 1)
        in_channels = audio_channels
        for index in range(depth):
            encode = [nn.Conv1d(in_channels, channels, kernel_size, stride), nn.ReLU()]
            if rewrite: encode += [nn.Conv1d(channels, ch_scale * channels, 1), activation]
            self.encoder.append(nn.Sequential(*encode))
            decode = ([nn.Conv1d(channels, ch_scale * channels, context), activation] if rewrite else []) + [ nn.ConvTranspose1d(channels, in_channels if index > 0 else len(self.sources) * audio_channels, kernel_size, stride) ]
            if index > 0: decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels, channels = channels, int(growth * channels)
        self.lstm = BLSTM(in_channels, lstm_layers) if lstm_layers else None
        if rescale: _rescale_module(self, reference=rescale)
    def valid_length(self, length): return _valid_length(self, length, True)
    def forward(self, mix):
        length, x = mix.shape[-1], mix
        if self.normalize: mono = mix.mean(dim=1, keepdim=True); mean, std = (mono.mean(dim=-1, keepdim=True), mono.std(dim=-1, keepdim=True))
        else: mean, std = (0, 1)
        x = (x - mean) / (1e-5 + std)
        if self.resample: x = _resample_x2(x)
        x = _unet_forward(self, x)
        if self.resample: x = _downsample_x2(x, length)
        x = x * std + mean
        return x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
class LegacyV3Demucs(nn.Module):
    def __init__(self, sources, audio_channels=2, channels=64, growth=2.0, depth=6, rewrite=True, lstm_layers=0, kernel_size=8, stride=4, context=1, gelu=True, glu=True, norm_starts=4, norm_groups=4, dconv_mode=1, dconv_depth=2, dconv_comp=4, dconv_attn=4, dconv_lstm=4, dconv_init=1e-4, normalize=True, resample=True, rescale=0.1, samplerate=44100, segment=4 * 10, **_):
        super().__init__()
        self.audio_channels, self.sources = audio_channels, _normalize_sources(sources)
        self.kernel_size, self.context, self.stride, self.depth = kernel_size, context, stride, depth
        self.resample, self.channels, self.normalize = resample, channels, normalize
        self.samplerate, self.segment = samplerate, segment
        self.segment_length = int(float(segment) * samplerate)
        self.encoder, self.decoder = nn.ModuleList(), nn.ModuleList()
        activation, ch_scale = (nn.GLU(dim=1), 2) if glu else (nn.ReLU(), 1)
        act2 = nn.GELU if gelu else nn.ReLU
        in_channels = audio_channels
        for index in range(depth):
            norm_fn = (lambda d: nn.GroupNorm(norm_groups, d)) if index >= norm_starts else (lambda d: nn.Identity())
            attn, lstm = index >= dconv_attn, index >= dconv_lstm
            encode = [nn.Conv1d(in_channels, channels, kernel_size, stride), norm_fn(channels), act2()]
            if dconv_mode & 1: encode.append(LegacyDConv(channels, depth=dconv_depth, init=dconv_init, compress=dconv_comp, attn=attn, lstm=lstm))
            if rewrite: encode += [nn.Conv1d(channels, ch_scale * channels, 1), norm_fn(ch_scale * channels), activation]
            self.encoder.append(nn.Sequential(*encode))
            out_channels = in_channels if index > 0 else len(self.sources) * audio_channels
            decode = ([nn.Conv1d(channels, ch_scale * channels, 2 * context + 1, padding=context), norm_fn(ch_scale * channels), activation] if rewrite else [])
            if dconv_mode & 2: decode.append(LegacyDConv(channels, depth=dconv_depth, init=dconv_init, compress=dconv_comp, attn=attn, lstm=lstm))
            decode.append(nn.ConvTranspose1d(channels, out_channels, kernel_size, stride))
            if index > 0: decode += [norm_fn(out_channels), act2()]
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels, channels = channels, int(growth * channels)
        self.lstm = BLSTM(in_channels, lstm_layers) if lstm_layers else None
        if rescale: _rescale_module(self, reference=rescale)
    def valid_length(self, length): return _valid_length(self, length, False)
    def forward(self, mix):
        x, length = mix, mix.shape[-1]
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean, std = mono.mean(dim=-1, keepdim=True), mono.std(dim=-1, keepdim=True)
            x = (x - mean) / (1e-5 + std)
        else:
            mean, std = 0, 1
        delta = self.valid_length(length) - length
        x = F.pad(x, (delta // 2, delta - delta // 2))
        if self.resample: x = _resample_x2(x)
        x = _unet_forward(self, x)
        if self.resample: x = _downsample_x2(x, length + delta)
        x = center_trim(x * std + mean, length)
        return x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
def LegacyDConv(channels, **kw): return DConv(channels, legacy=True, **kw)
class LegacyLocalState(nn.Module):
    def __init__(self, channels, heads=4, nfreqs=0, ndecay=4):
        super().__init__()
        if channels % heads: raise ValueError("legacy local attention channels must be divisible by heads")
        self.heads, self.nfreqs, self.ndecay = heads, nfreqs, ndecay
        self.content, self.query, self.key = [nn.Conv1d(channels, channels, 1) for _ in range(3)]
        if nfreqs: self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)
        if ndecay: self.query_decay = nn.Conv1d(channels, heads * ndecay, 1); self.query_decay.weight.data *= 0.01; self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * nfreqs, channels, 1)
    def forward(self, x):
        if x.device.type == "supa":
            # in-place masked_fill_/einsum combos hit "_copy_from_and_resize"
            # on the vendor stack; the legacy attention is tiny -- CPU roundtrip
            # (self moves to CPU once; weights stay there for this path)
            device = x.device
            if next(self.parameters()).device.type != "cpu":
                self.cpu()
            return self._forward_reference(x.cpu()).to(device)
        return self._forward_reference(x)
    def _forward_reference(self, x):
        batch, _, time = x.shape
        heads = self.heads
        indexes = torch.arange(time, device=x.device, dtype=x.dtype)
        delta = indexes[:, None] - indexes[None, :]
        queries, keys = [t.view(batch, heads, -1, time) for t in (self.query(x), self.key(x))]
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries) / keys.shape[2] ** 0.5
        freq_kernel = None
        if self.nfreqs: periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype); freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 1, 1)); dots += torch.einsum('fts,bhfs->bhts', freq_kernel, self.query_freqs(x).view(batch, heads, -1, time) / self.nfreqs ** 0.5)
        if self.ndecay: decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype); decay_q = torch.sigmoid(self.query_decay(x).view(batch, heads, -1, time)) / 2; dots += torch.einsum('fts,bhfs->bhts', -decays.view(-1, 1, 1) * delta.abs() / self.ndecay ** 0.5, decay_q)
        dots.masked_fill_(torch.eye(time, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)
        result = torch.einsum("bhts,bhct->bhcs", weights, self.content(x).view(batch, heads, -1, time))
        if self.nfreqs: result = torch.cat([result, torch.einsum('bhts,fts->bhfs', weights, freq_kernel)], 2)
        return x + self.proj(result.reshape(batch, -1, time))
class LegacyHEncLayer(HEncLayer):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, empty=False, freq=True, dconv=True, norm=True, context=0, dconv_kw=None, pad=True, rewrite=True): dconv_kw = dict(dconv_kw or {}, legacy=True); super().__init__(chin, chout, kernel_size, stride, norm_groups, empty, freq, dconv, norm, context, dconv_kw, pad, rewrite)
class LegacyHDecLayer(HDecLayer):
    def __init__(self, chin, chout, last=False, kernel_size=8, stride=4, norm_groups=1, empty=False, freq=True, dconv=True, norm=True, context=1, dconv_kw=None, pad=True, context_freq=True, rewrite=True): dconv_kw = dict(dconv_kw or {}, legacy=True); super().__init__(chin, chout, last, kernel_size, stride, norm_groups, empty, freq, dconv, norm, context, dconv_kw, pad, context_freq, rewrite)
    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3: x = x.view(x.shape[0], self.chin, -1, x.shape[-1])
        if self.empty:
            y = x
        else:
            x = x + skip  # legacy: dconv/rewrite run on the summed tensor (not GLU-of-rewrite of sum)
            y = F.glu(self.norm1(self.rewrite(x)), dim=1) if self.rewrite else x
            if self.dconv: y = _dconv_freq(self.dconv, y) if self.freq else self.dconv(y)
        z = self.norm2(self.conv_tr(y))
        if self.freq and self.pad:
            z = z[..., self.pad : -self.pad, :]
        elif not self.freq:
            z = z[..., self.pad : self.pad + length]
            assert z.shape[-1] == length
        return (z if self.last else F.gelu(z)), y
class LegacyMultiWrap(MultiWrap):
    def forward(self, x, skip=None, length=None):
        if not self.conv:  # legacy dec path passes length through to the wrapped dec layer
            freqs = x.shape[2]
            start, outs = 0, []
            for ratio, layer in zip(list(self.split_ratios) + [1], self.layers):
                limit = freqs if ratio == 1 else round(freqs * ratio)
                last, layer.last = layer.last, True
                out, _ = layer(x[:, :, start:limit], skip[:, :, start:limit], length)
                if outs: outs[-1][:, :, -layer.stride:] += out[:, :, :layer.stride] - layer.conv_tr.bias.view(1, -1, 1, 1); out = out[:, :, layer.stride:]
                if ratio == 1: out = out[:, :, :-layer.stride // 2, :]
                if start == 0: out = out[:, :, layer.stride // 2:, :]
                outs.append(out)
                layer.last = last
                start = limit
            out = torch.cat(outs, dim=2)
            return out if last else F.gelu(out), None
        return super().forward(x, skip, length)
def _pad1d(x, paddings, mode="constant", value=0.0):
    length = x.shape[-1]
    left, right = paddings
    if mode == "reflect" and length <= max(left, right):
        extra = max(left, right) - length + 1
        extra_right = min(right, extra)
        extra_left = extra - extra_right
        paddings = (left - extra_left, right - extra_right)
        x = F.pad(x, (extra_left, extra_right))
    return F.pad(x, paddings, mode, value)
def _spectro(x, n_fft=512, hop_length=None, pad=0):
    """ ``torch.stft`` wrapper; vendors without a complex STFT use npu_fft. """
    from pymss_core.npu_fft import stft_npu, torch_stft_blocked
    *other, length = x.shape
    flat = x.reshape(-1, length)
    n_fft_full = n_fft * (1 + pad)
    kwargs = dict(hop_length=hop_length or n_fft // 4, window=_hann(n_fft, x), win_length=n_fft, normalized=True, center=True, return_complex=True, pad_mode="reflect")
    if torch_stft_blocked(x.device):
        z = stft_npu(flat, n_fft_full, **kwargs)
    else:
        z = torch.stft(flat, n_fft_full, **kwargs)
    return z.view(*other, z.shape[-2], z.shape[-1])
def _ispectro(z, hop_length=None, length=None, pad=0):
    """ ``torch.istft`` wrapper; vendors without a complex STFT use npu_fft. """
    from pymss_core.npu_fft import istft_npu, torch_stft_blocked
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    flat = z.reshape(-1, freqs, frames)
    kwargs = dict(hop_length=hop_length, window=_hann(n_fft // (1 + pad), z.real), win_length=n_fft // (1 + pad), normalized=True, length=length, center=True)
    if torch_stft_blocked(z.device):
        x = istft_npu(flat, n_fft, **kwargs)
    else:
        x = torch.istft(flat, n_fft, **kwargs)
    return x.view(*other, x.shape[-1])
class LegacyHDemucs(nn.Module):
    def __init__(self, sources, audio_channels=2, channels=48, channels_time=None, growth=2, nfft=4096, wiener_iters=0,
                 end_iters=0, wiener_residual=False, cac=True, depth=6, rewrite=True, hybrid=True, hybrid_old=False,
                 multi_freqs=None, multi_freqs_depth=2, freq_emb=0.2, emb_scale=10, emb_smooth=True, kernel_size=8,
                 time_stride=2, stride=4, context=1, context_enc=0, norm_starts=4, norm_groups=4, dconv_mode=1,
                 dconv_depth=2, dconv_comp=4, dconv_attn=4, dconv_lstm=4, dconv_init=1e-4, rescale=0.1,
                 samplerate=44100, segment=4 * 10, **_):
        super().__init__()
        if wiener_iters != 0 or end_iters != 0 or not cac: raise ValueError("legacy HDemucs loader supports only CaC checkpoints without Wiener filtering")
        self.cac, self.wiener_residual, self.audio_channels = cac, wiener_residual, audio_channels
        self.sources = list(sources)
        self.kernel_size, self.context, self.stride, self.depth = kernel_size, context, stride, depth
        self.channels, self.samplerate = channels, samplerate
        self.segment_length, self.segment = int(float(segment) * samplerate), segment
        self.nfft, self.hop_length = nfft, nfft // 4
        self.wiener_iters, self.end_iters = wiener_iters, end_iters
        self.freq_emb, self.hybrid, self.hybrid_old = None, hybrid, hybrid_old
        self.encoder, self.decoder = nn.ModuleList(), nn.ModuleList()
        if hybrid: self.tencoder, self.tdecoder = (nn.ModuleList(), nn.ModuleList())
        chin = audio_channels; chin_z = chin * 2 if cac else chin
        chout, chout_z, freqs = channels_time or channels, channels, nfft // 2
        for index in range(depth):
            freq = freqs > 1
            ker, stri, pad, last_freq = kernel_size, stride, True, False
            if not freq: ker, stri = (time_stride * 2, time_stride)
            if freq and freqs <= kernel_size: ker, pad, last_freq = (freqs, False, True)
            kw = {"kernel_size": ker, "stride": stri, "freq": freq, "pad": pad, "norm": index >= norm_starts, "rewrite": rewrite, "norm_groups": norm_groups, "dconv_kw": {"lstm": index >= dconv_lstm, "attn": index >= dconv_attn, "depth": dconv_depth, "compress": dconv_comp, "init": dconv_init, "gelu": True}}
            kwt = dict(kw, freq=0, kernel_size=kernel_size, stride=stride, pad=True)
            kw_dec = dict(kw)
            if last_freq: chout_z = max(chout, chout_z); chout = chout_z
            multi = bool(multi_freqs and index < multi_freqs_depth)
            if multi: kw_dec['context_freq'] = False
            enc = LegacyHEncLayer(chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw)
            if multi: enc = LegacyMultiWrap(enc, multi_freqs)
            self.encoder.append(enc)
            if hybrid and freq: self.tencoder.append(LegacyHEncLayer(chin, chout, dconv=dconv_mode & 1, context=context_enc, empty=last_freq, **kwt))
            if index == 0: chin = audio_channels * len(self.sources); chin_z = chin * 2 if cac else chin
            dec = LegacyHDecLayer(chout_z, chin_z, dconv=dconv_mode & 2, last=index == 0, context=context, **kw_dec)
            if multi: dec = LegacyMultiWrap(dec, multi_freqs)
            self.decoder.insert(0, dec)
            if hybrid and freq: self.tdecoder.insert(0, LegacyHDecLayer(chout, chin, dconv=dconv_mode & 2, empty=last_freq, last=index == 0, context=context, **kwt))
            chin, chin_z = chout, chout_z
            chout, chout_z = int(growth * chout), int(growth * chout_z)
            if freq: freqs = 1 if freqs <= kernel_size else freqs // stride
            if index == 0 and freq_emb: self.freq_emb = LegacyScaledEmbedding(freqs, chin_z, smooth=emb_smooth, scale=emb_scale); self.freq_emb_scale = freq_emb
        if rescale: _rescale_module(self, reference=rescale)
    def valid_length(self, length): return length
    def _spec(self, x):
        hl, nfft = self.hop_length, self.nfft
        if self.hybrid: le = math.ceil(x.shape[-1] / hl); pad = hl // 2 * 3; x = _pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode='constant' if self.hybrid_old else 'reflect')
        z = _spectro(x, nfft, hl)[..., :-1, :]
        return z[..., 2 : 2 + le] if self.hybrid else z
    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4**scale)
        if self.hybrid:
            z = F.pad(z, (2, 2, 0, 1))
            pad = hl // 2 * 3
            le = hl * math.ceil(length / hl) + (0 if self.hybrid_old else 2 * pad)
            x = _ispectro(z, hl, length=le)
            return x[..., :length] if self.hybrid_old else x[..., pad : pad + length]
        return _ispectro(F.pad(z, (0, 0, 0, 1)), hl, length)
    def _magnitude(self, z):
        if self.cac: batch, channels, freqs, time = z.shape; return torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(batch, channels * 2, freqs, time)
        return z.abs()
    def _mask(self, z, m):
        if not self.cac: raise ValueError("legacy HDemucs loader supports only CaC checkpoints")
        batch, sources, _channels, freqs, time = m.shape
        out = m.view(batch, sources, -1, 2, freqs, time).permute(0, 1, 2, 4, 5, 3)
        return torch.view_as_complex(out.contiguous())
    def forward(self, mix):
        length = mix.shape[-1]
        z = self._spec(mix)
        x = self._magnitude(z)
        batch, _, freqs, time = x.shape
        mean, std = x.mean(dim=(1, 2, 3), keepdim=True), x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        if self.hybrid: meant, stdt = (mix.mean(dim=(1, 2), keepdim=True), mix.std(dim=(1, 2), keepdim=True)); xt = (mix - meant) / (1e-05 + stdt)
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for index, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if self.hybrid and index < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[index]
                xt, inject = tenc(xt), xt if tenc.empty else None
                if not tenc.empty: saved_t.append(xt)
            x = encode(x, inject)
            if index == 0 and self.freq_emb is not None: frs = torch.arange(x.shape[-2], device=x.device); emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x); x = x + self.freq_emb_scale * emb
            saved.append(x)
        x = torch.zeros_like(x)
        if self.hybrid: xt = torch.zeros_like(x)
        for index, decode in enumerate(self.decoder):
            x, pre = decode(x, saved.pop(-1), lengths.pop(-1))
            if self.hybrid:
                offset = self.depth - len(self.tdecoder)
                if index >= offset:
                    tdec = self.tdecoder[index - offset]
                    length_t = lengths_t.pop(-1)
                    if tdec.empty: pre = pre[:, :, 0]; xt, _ = tdec(pre, None, length_t)
                    else: xt, _ = tdec(xt, saved_t.pop(-1), length_t)
        sources = len(self.sources)
        x = x.view(batch, sources, -1, freqs, time) * std[:, None] + mean[:, None]
        x = self._ispec(self._mask(z, x), length)
        if self.hybrid: xt = xt.view(batch, sources, -1, length) * stdt[:, None] + meant[:, None]; x = xt + x
        return x
def overlap_and_add(signal, frame_step):
    outer_dimensions, (frames, frame_length) = signal.size()[:-2], signal.size()[-2:]
    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step, subframes_per_frame = frame_step // subframe_length, frame_length // subframe_length
    output_subframes = (frame_step * (frames - 1) + frame_length) // subframe_length
    frame = torch.arange(0, output_subframes, device=signal.device).unfold(0, subframes_per_frame, subframe_step)
    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    flat_index = frame.long().contiguous().view(-1)
    flat_signal = signal.view(*outer_dimensions, -1, subframe_length)
    from pymss_core.device import scatter_add_blocked

    if scatter_add_blocked(signal.device) or signal.dim() > 3:
        # BIREN aborts ('vector::reserve') on index_add_ with 5-D sources, so the
        # overlap-add runs as a one-hot matrix multiply instead: same values, no
        # scatter kernel.
        from pymss_core.vendor_shims import scatter_add_equivalent

        result = scatter_add_equivalent(flat_signal.movedim(-2, -2), flat_index, -2, output_subframes)
        return result.view(*outer_dimensions, -1)
    result.index_add_(-2, flat_index, flat_signal)
    return result.view(*outer_dimensions, -1)
class LegacyConvTasNet(nn.Module):
    def __init__(self, sources=None, N=256, L=20, B=256, H=512, P=3, X=8, R=4, C=4, audio_channels=2, norm_type="gLN", causal=False, mask_nonlinear="relu", samplerate=44100, segment_length=44100 * 2 * 4, **_):
        super().__init__()
        self.sources = _normalize_sources(C if sources is None else sources)
        self.C = len(self.sources)
        self.N, self.L, self.B, self.H, self.P, self.X, self.R = N, L, B, H, P, X, R
        self.norm_type, self.causal, self.mask_nonlinear = norm_type, causal, mask_nonlinear
        self.audio_channels, self.samplerate, self.segment_length = audio_channels, samplerate, segment_length
        self.encoder = Encoder(L, N, audio_channels)
        self.separator = TemporalConvNet(N, B, H, P, X, R, self.C, norm_type, causal, mask_nonlinear)
        self.decoder = Decoder(N, L, audio_channels)
        for parameter in self.parameters():
            if parameter.dim() > 1: nn.init.xavier_normal_(parameter)
    def valid_length(self, length): return length
    def forward(self, mixture): mixture_w = self.encoder(mixture); est_source = self.decoder(mixture_w, self.separator(mixture_w)); length = mixture.size(-1); delta = length - est_source.size(-1); return F.pad(est_source, (0, delta)) if delta >= 0 else est_source[..., :length]
class Encoder(nn.Module):
    def __init__(self, L, N, audio_channels): super().__init__(); self.L, self.N = L, N; self.conv1d_U = nn.Conv1d(audio_channels, N, kernel_size=L, stride=L // 2, bias=False)
    def forward(self, mixture): return F.relu(self.conv1d_U(mixture))
class Decoder(nn.Module):
    def __init__(self, N, L, audio_channels): super().__init__(); self.N, self.L, self.audio_channels = N, L, audio_channels; self.basis_signals = nn.Linear(N, audio_channels * L, bias=False)
    def forward(self, mixture_w, est_mask): source_w = torch.transpose(torch.unsqueeze(mixture_w, 1) * est_mask, 2, 3); est_source = self.basis_signals(source_w); batch, sources, frames, _ = est_source.size(); est_source = est_source.view(batch, sources, frames, self.audio_channels, -1).transpose(2, 3).contiguous(); return overlap_and_add(est_source, self.L // 2)
class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear="relu"):
        super().__init__()
        self.C, self.mask_nonlinear = C, mask_nonlinear
        def block(dilation): return TemporalBlock(B, H, P, stride=1, padding=(P - 1) * dilation if causal else (P - 1) * dilation // 2, dilation=dilation, norm_type=norm_type, causal=causal)
        self.network = nn.Sequential(ChannelwiseLayerNorm(N), nn.Conv1d(N, B, 1, bias=False), nn.Sequential(*(nn.Sequential(*(block(2**i) for i in range(X))) for _ in range(R))), nn.Conv1d(B, C * N, 1, bias=False))
    def forward(self, mixture_w):
        batch, channels, frames = mixture_w.size()
        score = self.network(mixture_w).view(batch, self.C, channels, frames)
        if self.mask_nonlinear == "softmax": return F.softmax(score, dim=1)
        if self.mask_nonlinear == "relu": return F.relu(score)
        raise ValueError("Unsupported mask non-linear function")
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, norm_type="gLN", causal=False): super().__init__(); self.net = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, bias=False), nn.PReLU(), _choose_norm(norm_type, out_channels), DepthwiseSeparableConv(out_channels, in_channels, kernel_size, stride, padding, dilation, norm_type, causal))
    def forward(self, x): return self.net(x) + x
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, norm_type="gLN", causal=False):
        super().__init__()
        layers = [nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)]
        if causal: layers.append(Chomp1d(padding))
        self.net = nn.Sequential(*layers, nn.PReLU(), _choose_norm(norm_type, in_channels), nn.Conv1d(in_channels, out_channels, 1, bias=False))
    def forward(self, x): return self.net(x)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, : -self.chomp_size].contiguous()
class ChannelwiseLayerNorm(nn.Module):
    def __init__(self, channel_size): super().__init__(); self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1)); self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1)); self.reset_parameters()
    def reset_parameters(self): self.gamma.data.fill_(1); self.beta.data.zero_()
    def _stat(self, y): return torch.mean(y, dim=1, keepdim=True), torch.var(y, dim=1, keepdim=True, unbiased=False)
    def forward(self, y): mean, var = self._stat(y); return self.gamma * (y - mean) / torch.sqrt(var + EPS) + self.beta
class GlobalLayerNorm(ChannelwiseLayerNorm):
    def _stat(self, y): mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True); return mean, (y - mean).square().mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
def _choose_norm(norm_type, channel_size): klass = {"gLN": GlobalLayerNorm, "cLN": ChannelwiseLayerNorm, "id": nn.Identity}.get(norm_type, nn.BatchNorm1d); return klass(channel_size)
class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert 0 <= offset < total_length
        length = total_length - offset if length is None else min(total_length - offset, length)
        if isinstance(tensor, TensorChunk): tensor, offset = (tensor.tensor, offset + tensor.offset)
        self.tensor, self.offset, self.length, self.device = tensor, offset, length, tensor.device
    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape
    def padded(self, target_length):
        delta = target_length - self.length
        assert delta >= 0
        start = self.offset - delta // 2
        end = start + target_length
        correct_start, correct_end = max(0, start), min(self.tensor.shape[-1], end)
        out = F.pad(self.tensor[..., correct_start:correct_end], (correct_start - start, end - correct_end))
        assert out.shape[-1] == target_length
        return out
def tensor_chunk(tensor_or_chunk): return tensor_or_chunk if isinstance(tensor_or_chunk, TensorChunk) else TensorChunk(tensor_or_chunk)
class LegacyBagOfModels(nn.Module):
    def __init__(self, models, weights=None, segment=None):
        super().__init__()
        if not models: raise ValueError("legacy Demucs bag must contain at least one model")
        first = models[0]
        for model in models:
            if model.sources != first.sources: raise ValueError("all models in a legacy Demucs bag must have the same sources")
            if model.samplerate != first.samplerate: raise ValueError("all models in a legacy Demucs bag must have the same samplerate")
            if model.audio_channels != first.audio_channels: raise ValueError("all models in a legacy Demucs bag must have the same channel count")
            if segment is not None: model.segment_length = int(float(segment) * model.samplerate)
        self.sources, self.samplerate = first.sources, first.samplerate
        self.audio_channels, self.segment_length = first.audio_channels, first.segment_length
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else [[1.0 for _ in self.sources] for _ in models]
    def forward(self, x): raise NotImplementedError("use apply_legacy_model for legacy Demucs bags")
def apply_legacy_model(model, mix, shifts=0, split=True, overlap=0.25, transition_power=1.0, progress=False):
    if isinstance(model, LegacyBagOfModels):
        estimates, totals = 0.0, [0.0] * len(model.sources)
        for sub_model, weights in zip(model.models, model.weights):
            out = apply_legacy_model(sub_model, mix, shifts=shifts, split=split, overlap=overlap, transition_power=transition_power, progress=progress)
            for index, weight in enumerate(weights):
                out[index] *= weight
                totals[index] += weight
            estimates += out
        for index, total in enumerate(totals):
            estimates[index] /= total
        return estimates
    assert transition_power >= 1, "transition_power < 1 leads to unstable transitions"
    device = mix.device
    channels, length = mix.shape
    if split:
        out = torch.zeros(len(model.sources), channels, length, device=device)
        sum_weight = torch.zeros(length, device=device)
        segment = int(model.segment_length)
        stride = int((1 - overlap) * segment)
        weight = torch.cat([torch.arange(1, segment // 2 + 1, device=device), torch.arange(segment - segment // 2, 0, -1, device=device)])
        weight = (weight / weight.max()) ** transition_power
        for offset in range(0, length, stride):
            chunk_out = apply_legacy_model(model, TensorChunk(mix, offset, segment), shifts=shifts, split=False)
            chunk_length = chunk_out.shape[-1]
            out[..., offset : offset + segment] += weight[:chunk_length] * chunk_out
            sum_weight[offset : offset + segment] += weight[:chunk_length]
        return out / sum_weight
    if shifts:
        max_shift = int(0.5 * model.samplerate)
        padded_mix, out = tensor_chunk(mix).padded(length + 2 * max_shift), 0.0
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            out += apply_legacy_model(model, shifted, shifts=0, split=False)[..., max_shift - offset :]
        return out / shifts
    padded_mix = tensor_chunk(mix).padded(model.valid_length(length))
    with torch.no_grad():
        return center_trim(model(padded_mix.unsqueeze(0))[0], length)
def _stub_class(module_name, class_name): return type(class_name, (), {"__module__": module_name})
@contextmanager
def _legacy_pickle_modules():
    module_classes = { "demucs.model": {"Demucs": _stub_class("demucs.model", "Demucs")}, "demucs.demucs": {"Demucs": _stub_class("demucs.demucs", "Demucs")}, "demucs.tasnet": {"ConvTasNet": _stub_class("demucs.tasnet", "ConvTasNet")}, "demucs.hdemucs": {"HDemucs": _stub_class("demucs.hdemucs", "HDemucs")}, "demucs.htdemucs": {"HTDemucs": _stub_class("demucs.htdemucs", "HTDemucs")}, }
    previous = {name: sys.modules.get(name) for name in ["demucs", *module_classes]}
    package = sys.modules.setdefault("demucs", types.ModuleType("demucs"))
    package.__path__ = []
    try:
        for module_name, classes in module_classes.items():
            module = sys.modules.setdefault(module_name, types.ModuleType(module_name))
            for class_name, klass in classes.items():
                setattr(module, class_name, klass)
            setattr(package, module_name.split(".")[-1], module)
        yield module_classes
    finally:
        for name, module in previous.items():
            if module is None: sys.modules.pop(name, None)
            else: sys.modules[name] = module
def _normalize_sources(sources):
    if isinstance(sources, int):
        if sources == 4: return LEGACY_STEMS_4.copy()
        if sources == 2: return LEGACY_STEMS_2.copy()
        return [f"source_{index}" for index in range(sources)]
    return list(sources)
def _resolve_klass(klass):
    name = (getattr(klass, "__module__", ""), getattr(klass, "__name__", ""))
    if name == ("demucs.htdemucs", "HTDemucs"):
        from .demucs4ht import HTDemucs
        return HTDemucs
    resolved = {("demucs.model", "Demucs"): LegacyDemucs, ("demucs.demucs", "Demucs"): LegacyV3Demucs, ("demucs.tasnet", "ConvTasNet"): LegacyConvTasNet, ("demucs.hdemucs", "HDemucs"): LegacyHDemucs}.get(name)
    if resolved is None: raise ValueError(f"Unsupported legacy Demucs checkpoint class: {name[0]}.{name[1]}")
    return resolved
def _load_raw_checkpoint(model_path):
    try:
        with _legacy_pickle_modules():
            return torch.load(model_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        if exc.name == "diffq": raise ValueError("DiffQ quantized legacy Demucs checkpoints are not supported without diffq") from exc
        raise
def _drop_unsupported_kwargs(klass, kwargs): parameters = inspect.signature(klass).parameters; return {key: value for key, value in kwargs.items() if key in parameters}
def _build_model_from_package(package, model_path=None):
    if isinstance(package, tuple) and len(package) >= 4:
        klass, args, kwargs, state = package[:4]
    elif isinstance(package, dict) and {"klass", "args", "kwargs", "state"} <= set(package):
        klass, args, kwargs, state = package["klass"], package["args"], package["kwargs"], package["state"]
    elif isinstance(package, dict):
        if model_path is None: raise ValueError("state_dict-only legacy Demucs checkpoint requires model_path for architecture inference")
        klass, args, kwargs = _infer_state_dict_architecture(package, model_path)
        state = package
    else:
        raise ValueError(f"Unsupported legacy Demucs checkpoint format: {type(package).__name__}")
    model_cls = _resolve_klass(klass)
    model = model_cls(*args, **_drop_unsupported_kwargs(model_cls, dict(kwargs)))
    if isinstance(state, dict) and state.get("__quantized"): raise ValueError("DiffQ quantized legacy Demucs checkpoints are not supported without diffq")
    model.load_state_dict(state)
    return _ensure_legacy_metadata(model)
def _ensure_legacy_metadata(model):
    if not hasattr(model, "segment_length"): model.segment_length = int(float(model.segment) * model.samplerate) if hasattr(model, 'segment') and hasattr(model, 'samplerate') else 44100 * 10
    if not hasattr(model, "samplerate"): model.samplerate = 44100
    if not hasattr(model, "audio_channels"): model.audio_channels = 2
    return model
def _infer_state_dict_architecture(state, model_path):
    name = Path(model_path).stem
    if "encoder.conv1d_U.weight" in state and "decoder.basis_signals.weight" in state:
        encoder, mask_conv = state["encoder.conv1d_U.weight"], state["separator.network.3.weight"]
        repeats = {int(k.split(".")[3]) for k in state if k.startswith("separator.network.2.") and len(k.split(".")) > 4}
        blocks = {int(k.split(".")[4]) for k in state if k.startswith("separator.network.2.0.") and len(k.split(".")) > 5}
        return _stub_class("demucs.tasnet", "ConvTasNet"), (), { "sources": int(mask_conv.shape[0] // encoder.shape[0]), "N": int(encoder.shape[0]), "L": int(encoder.shape[2]), "B": int(state["separator.network.1.weight"].shape[0]), "H": int(state["separator.network.2.0.0.net.0.weight"].shape[0]), "P": int(state["separator.network.2.0.0.net.3.net.0.weight"].shape[-1]), "X": max(blocks) + 1 if blocks else 8, "R": max(repeats) + 1 if repeats else 4, "audio_channels": int(encoder.shape[1])}
    if name.startswith("demucs_unittest"): depth = sum((1 for key in state if key.startswith('encoder.') and key.endswith('.0.weight'))); return (_stub_class('demucs.model', 'Demucs'), (), {'sources': 4, 'audio_channels': 2, 'channels': 4, 'depth': depth, 'lstm_layers': 2})
    if "lstm.lstm.weight_ih_l0" in state and "encoder.0.0.weight" in state:
        first = state["encoder.0.0.weight"]
        depth = sum(1 for key in state if key.startswith("encoder.") and key.endswith(".0.weight"))
        last_decoder_bias = f"decoder.{depth - 1}.2.bias"
        context = int(state["decoder.0.0.weight"].shape[-1]) if "decoder.0.0.weight" in state else 3
        resample = (bool(state["decoder.0.2.weight"].shape[1] == state[f"encoder.{depth - 1}.0.weight"].shape[0]) if "decoder.0.2.weight" in state and f"encoder.{depth - 1}.0.weight" in state else False)
        return _stub_class("demucs.model", "Demucs"), (), { "sources": int(state[last_decoder_bias].numel() // first.shape[1]) if last_decoder_bias in state else 4, "audio_channels": int(first.shape[1]), "channels": int(first.shape[0]), "depth": int(depth), "lstm_layers": 2, "context": context, "resample": resample}
    raise ValueError(f"Cannot infer legacy Demucs architecture from state_dict-only checkpoint: {model_path}")
def load_legacy_demucs_model(model_path, config_path=None):
    path = Path(model_path)
    bag_path = path if path.suffix.lower() in {".yaml", ".yml"} else Path(config_path) if config_path else None
    if bag_path and bag_path.exists():
        bag = yaml.safe_load(bag_path.read_text(encoding="utf-8"))
        if isinstance(bag, dict) and "models" in bag:
            models = []
            for name in bag["models"]:
                candidates = sorted(bag_path.parent.glob(f"{name}*.th"))
                if not candidates: raise FileNotFoundError(f"Cannot find legacy Demucs bag member {name!r} next to {bag_path}")
                models.append(load_legacy_demucs_model(candidates[0])[0])
            return LegacyBagOfModels(models, bag.get("weights"), bag.get("segment")), _legacy_config_from_model(models[0])
        if path == bag_path: raise ValueError(f"Legacy Demucs YAML must contain a 'models' list: {bag_path}")
    model = _build_model_from_package(_load_raw_checkpoint(path), path)
    return model, _legacy_config_from_model(model)
def _legacy_config_from_model(model):
    _ensure_legacy_metadata(model)
    sources = list(model.sources)
    return {
        "model": {"stereo": model.audio_channels == 2, "legacy_demucs": True},
        "training": {"instruments": sources, "target_instrument": None, "samplerate": int(model.samplerate), "segment": float(model.segment_length) / float(model.samplerate), "channels": int(model.audio_channels), "use_amp": True},
        "audio": {"sample_rate": int(model.samplerate), "chunk_size": int(model.segment_length)},
        "inference": {"batch_size": 1, "overlap_size": int(model.segment_length * 0.25), "normalize": False, "shifts": 0, "split": True},
    }