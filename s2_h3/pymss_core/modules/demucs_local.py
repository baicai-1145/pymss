import math
import random
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pymss_core.npu_fft import istft_npu as _npu_fft_istft, stft_npu as _npu_fft_stft, torch_stft_blocked
_HANN_CACHE = {}


def _hann(n_fft, ref):  # hann_window is a pure function of n_fft: memoize per (len, device, dtype)
    key = (n_fft, ref.device, ref.dtype)
    w = _HANN_CACHE.get(key)
    if w is None:
        w = torch.hann_window(n_fft, device="cpu", dtype=ref.dtype).to(ref) if ref.device.type == "npu" else torch.hann_window(n_fft).to(ref)
        _HANN_CACHE[key] = w
    return w
def pad1d(x, paddings, mode="constant", value=0.0):
    length, (left, right) = x.shape[-1], paddings
    if mode == "reflect" and length <= max(left, right):  # torch reflect pad breaks past half window: overpad then crop
        extra = max(left, right) - length + 1
        extra_right = min(right, extra)
        extra_left = extra - extra_right
        paddings, x = (left - extra_left, right - extra_right), F.pad(x, (extra_left, extra_right))
    return F.pad(x, paddings, mode, value)
def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    if x.device.type == "mps": x = x.cpu()
    fn = _npu_fft_stft if torch_stft_blocked(x.device) else torch.stft
    z = fn(x.reshape(-1, length), n_fft * (1 + pad), hop_length or n_fft // 4, window=_hann(n_fft, x), win_length=n_fft, normalized=True, center=True, return_complex=True, pad_mode="reflect")
    return z.view(*other, z.shape[-2], z.shape[-1])
def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    win_length = n_fft // (1 + pad)
    if z.device.type == "mps": z = z.cpu()
    fn = _npu_fft_istft if torch_stft_blocked(z.device) else torch.istft
    x = fn(z.reshape(-1, freqs, frames), n_fft, hop_length, window=_hann(win_length, z.real), win_length=win_length, normalized=True, length=length, center=True)
    return x.view(*other, x.shape[-1])
def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            scale = (sub.weight.std().detach() / reference) ** 0.5
            sub.weight.data /= scale
            if sub.bias is not None: sub.bias.data /= scale
def _freq_dconv(dconv, y): b, c, fr, t = y.shape; return dconv(y.permute(0, 2, 1, 3).reshape(-1, c, t)).view(b, fr, c, t).transpose(1, 2)
def _raw_lstm_cpu(seq, weights, bias, num_layers, bidirectional):
    """Reference LSTM loop; torch_br hooks aten::lstm even for CPU tensors."""
    layer_input = seq
    idx = 0
    for _layer in range(num_layers):
        outputs = []
        for direction in (0, 1) if bidirectional else (0,):
            w_ih, w_hh = weights[idx].t().contiguous(), weights[idx + 1].t().contiguous()
            b_ih = b_hh = None
            if bias:
                b_ih, b_hh = weights[idx + 2], weights[idx + 3]
            idx += 4 if bias else 2
            x = layer_input.flip(0) if direction else layer_input
            hidden = w_ih.shape[1] // 4
            h = x.new_zeros(x.shape[1], hidden)
            c = torch.zeros_like(h)
            steps = []
            for t in range(x.shape[0]):
                gates = x[t] @ w_ih
                hg = h @ w_hh
                if b_ih is not None:
                    gates = gates + b_ih
                    hg = hg + b_hh
                gates = gates + hg
                i, f, g, o = gates.chunk(4, dim=-1)
                c = f.sigmoid() * c + i.sigmoid() * g.tanh()
                h = o.sigmoid() * c.tanh()
                steps.append(h)
            out = torch.stack(steps, dim=0)
            if direction:
                out = out.flip(0)
            outputs.append(out)
        layer_input = torch.cat(outputs, dim=-1) if bidirectional else outputs[0]
    return layer_input


class BLSTM(nn.Module):
    def __init__(self, dim, layers=1, max_steps=None, skip=False): super().__init__(); self.max_steps = max_steps; self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim); self.linear = nn.Linear(2 * dim, dim); self.skip = skip
    def forward(self, x):
        # The vendor fused-LSTM kernel drifts the time length on SUPA (skip-add
        # broadcast 800 vs 860) and torch_br hooks aten::lstm/mm even for CPU
        # tensors; the manual reference loop below runs on the input device with
        # element ops + small matmuls that all work, and has no length drift.
        if x.device.type == "supa":
            return self._forward_reference(x)
        return self._forward_torch(x)
    def _forward_reference(self, x):
        lstm = self.lstm

        def run(batch):  # (N, C, T) -> (N, C, T) through manual LSTM + linear
            seq = batch.permute(2, 0, 1)
            weights = [w.detach() for w in list(lstm._flat_weights)]
            out = _raw_lstm_cpu(seq, weights, lstm.bias, lstm.num_layers, lstm.bidirectional)
            return self.linear(out).permute(1, 2, 0)

        y, framed = x, False
        if self.max_steps is not None and x.shape[-1] > self.max_steps:
            batch, channels, length = x.shape
            width, stride = self.max_steps, self.max_steps // 2
            covered = width + stride * -(-(length - width) // stride)  # ceil
            if covered > length:
                x = torch.nn.functional.pad(x, (0, covered - length))
            frames = x.unfold(-1, width, stride)
            nframes, framed = frames.shape[2], True
            x = run(frames.permute(0, 2, 1, 3).reshape(-1, channels, width))
        else:
            x = run(x)
        if framed:
            frames = x.reshape(batch, -1, channels, width); limit = stride // 2
            pieces = []
            for i in range(nframes):
                lo = limit if i else 0
                hi = width if not (i < nframes - 1 or not i) else width - limit
                pieces.append(frames[:, i, :, lo:hi])
            x = torch.cat(pieces, -1)[..., :length]
        return x + y if self.skip else x
    def _forward_torch(self, x):
        y, framed = x, False
        if self.max_steps is not None and x.shape[-1] > self.max_steps:
            batch, channels, length = x.shape
            width, stride = self.max_steps, self.max_steps // 2
            covered = width + stride * -(-(length - width) // stride)  # ceil
            if covered > length:
                x = torch.nn.functional.pad(x, (0, covered - length))
            frames = x.unfold(-1, width, stride)
            nframes, framed = frames.shape[2], True
            flat = frames.permute(0, 2, 1, 3).reshape(-1, channels, width)
            x = self.linear(self.lstm(flat.permute(2, 0, 1))[0]).permute(1, 2, 0)
        else:
            x = self.linear(self.lstm(x.permute(2, 0, 1))[0]).permute(1, 2, 0)
        if framed:
            stacked = x.reshape(batch, -1, channels, width); limit = stride // 2
            pieces = []
            for i in range(nframes):
                lo = limit if i else 0
                hi = width if not (i < nframes - 1 or not i) else width - limit
                pieces.append(stacked[:, i, :, lo:hi])
            x = torch.cat(pieces, -1)[..., :length]
        return x + y if self.skip else x
class LegacyLayerScale(nn.Module):
    def __init__(self, channels, init=0): super().__init__(); self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True)); self.scale.data[:] = init
    def forward(self, x): return self.scale[:, None] * x
class LayerScale(nn.Module):
    def __init__(self, channels, init=0, channel_last=False): super().__init__(); self.channel_last = channel_last; self.scale = nn.Parameter(torch.full((channels,), init))
    def forward(self, x): return self.scale * x if self.channel_last else self.scale[:, None] * x
class DConv(nn.Module):
    def __init__(self, channels, compress=4, depth=2, init=1e-4, norm=True, gelu=True, kernel=3, legacy=False, attn=False, heads=4, ndecay=4, lstm=False, **_):
        super().__init__()
        hidden, self.legacy = int(channels / compress), legacy
        norm_fn = (lambda d: nn.GroupNorm(1, d)) if norm else (lambda d: nn.Identity())
        scale_cls = (lambda c: LegacyLayerScale(c, init)) if legacy else (lambda c: LayerScale(c, init))
        self.layers = nn.ModuleList()
        for d in range(abs(depth)):
            dilation = 2**d if depth > 0 else 1
            mods = [nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=dilation * (kernel // 2)), norm_fn(hidden), (nn.GELU if gelu else nn.ReLU)(), nn.Conv1d(hidden, 2 * channels, 1), norm_fn(2 * channels), nn.GLU(1), scale_cls(channels)]
            if legacy and attn:
                from .legacy_demucs import LegacyLocalState
                mods.insert(3, LegacyLocalState(hidden, heads=heads, ndecay=ndecay))
            if legacy and lstm: mods.insert(3, BLSTM(hidden, layers=2, max_steps=200, skip=True))
            self.layers.append(nn.Sequential(*mods))
    def forward(self, x):
        for layer in self.layers: x = x + layer(x)
        return x
class ScaledEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, scale=10.0, smooth=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:  # low-pass the table along the freq axis
            w = torch.cumsum(self.embedding.weight.data, dim=0)
            self.embedding.weight.data[:] = w / torch.arange(1, num_embeddings + 1).to(w).sqrt()[:, None]
        self.embedding.weight.data /= scale
        self.scale = scale
    @property
    def weight(self): return self.embedding.weight * self.scale
    def forward(self, x): return self.embedding(x) * self.scale
class HEncLayer(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, empty=False, freq=True, dconv=True, norm=True, context=0, dconv_kw=None, pad=True, rewrite=True):
        super().__init__()
        norm_fn = (lambda d: nn.GroupNorm(norm_groups, d)) if norm else (lambda d: nn.Identity())
        pad = kernel_size // 4 if pad else 0
        klass = nn.Conv2d if freq else nn.Conv1d
        self.freq, self.kernel_size, self.stride, self.empty, self.norm, self.pad = freq, kernel_size, stride, empty, norm, pad
        if freq:  # B,C,F,T conv: stride along freq only
            kernel_size, stride, pad = [kernel_size, 1], [stride, 1], [pad, 0]
        self.conv = klass(chin, chout, kernel_size, stride, pad)
        if empty: return
        self.norm1 = norm_fn(chout)
        self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context) if rewrite else None
        if rewrite: self.norm2 = norm_fn(2 * chout)
        self.dconv = DConv(chout, **(dconv_kw or {})) if dconv else None
    def forward(self, x, inject=None):
        if not self.freq and x.dim() == 4: x = x.view(x.shape[0], -1, x.shape[-1])
        if not self.freq and x.shape[-1] % self.stride: x = F.pad(x, (0, self.stride - x.shape[-1] % self.stride))
        y = self.conv(x)
        if self.empty: return y
        if inject is not None: y = y + (inject[:, :, None] if inject.dim() == 3 and y.dim() == 4 else inject)
        y = F.gelu(self.norm1(y))
        if self.dconv: y = _freq_dconv(self.dconv, y) if self.freq else self.dconv(y)
        return F.glu(self.norm2(self.rewrite(y)), dim=1) if self.rewrite else y
class HDecLayer(nn.Module):
    def __init__(self, chin, chout, last=False, kernel_size=8, stride=4, norm_groups=1, empty=False, freq=True, dconv=True, norm=True, context=1, dconv_kw=None, pad=True, context_freq=True, rewrite=True):
        super().__init__()
        norm_fn = (lambda d: nn.GroupNorm(norm_groups, d)) if norm else (lambda d: nn.Identity())
        self.pad = kernel_size // 4 if pad else 0
        self.last, self.freq, self.chin, self.empty, self.stride = last, freq, chin, empty, stride
        self.kernel_size, self.norm, self.context_freq = kernel_size, norm, context_freq
        klass, klass_tr = (nn.Conv2d, nn.ConvTranspose2d) if freq else (nn.Conv1d, nn.ConvTranspose1d)
        k, s = ([kernel_size, 1], [stride, 1]) if freq else (kernel_size, stride)
        self.conv_tr = klass_tr(chin, chout, k, s)
        self.norm2 = norm_fn(chout)
        if empty: return
        self.rewrite = None
        if rewrite: self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context) if context_freq else klass(chin, 2 * chin, [1, 1 + 2 * context], 1, [0, context]); self.norm1 = norm_fn(2 * chin)
        self.dconv = DConv(chin, **(dconv_kw or {})) if dconv else None
    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3: x = x.view(x.shape[0], self.chin, -1, x.shape[-1])
        if self.empty:
            y = x
        else:
            y = F.glu(self.norm1(self.rewrite(x + skip)), dim=1) if self.rewrite else x + skip
            if self.dconv: y = _freq_dconv(self.dconv, y) if self.freq else self.dconv(y)
        z = self.norm2(self.conv_tr(y))
        if self.freq and self.pad:
            z = z[..., self.pad : -self.pad, :]
        elif not self.freq:
            z = z[..., self.pad : self.pad + length]
            assert z.shape[-1] == length
        return (z if self.last else F.gelu(z)), y
class MultiWrap(nn.Module):
    def __init__(self, layer, split_ratios):
        super().__init__()
        self.split_ratios, self.conv = split_ratios, isinstance(layer, HEncLayer)
        def new_layer():
            lay = deepcopy(layer)
            if self.conv: lay.conv.padding = (0, 0)
            else: lay.pad = False
            for m in lay.modules():
                if hasattr(m, "reset_parameters"): m.reset_parameters()
            return lay
        self.layers = nn.ModuleList([new_layer() for _ in range(len(split_ratios) + 1)])
    def forward(self, x, skip=None, length=None):
        _, _, fr, _ = x.shape
        start, outs = 0, []
        for ratio, layer in zip(list(self.split_ratios) + [1], self.layers):
            if self.conv:
                pad, limit = layer.kernel_size // 4, fr if ratio == 1 else round(fr * ratio)
                if ratio != 1: le = limit - start + (pad if start == 0 else 0); limit = start + (round((le - layer.kernel_size) / layer.stride + 1) - 1) * layer.stride + layer.kernel_size - (pad if start == 0 else 0)
                y = x[:, :, start:limit, :]
                if start == 0: y = F.pad(y, (0, 0, pad, 0))
                if ratio == 1: y = F.pad(y, (0, 0, 0, pad))
                outs.append(layer(y))
                start = limit - layer.kernel_size + layer.stride
            else:
                limit, last = fr if ratio == 1 else round(fr * ratio), layer.last
                layer.last = True
                out, _ = layer(x[:, :, start:limit], skip[:, :, start:limit], None)
                if outs: outs[-1][:, :, -layer.stride:] += out[:, :, :layer.stride] - layer.conv_tr.bias.view(1, -1, 1, 1); out = out[:, :, layer.stride:]
                if ratio == 1: out = out[:, :, :-layer.stride // 2, :]
                if start == 0: out = out[:, :, layer.stride // 2:, :]
                outs.append(out)
                layer.last = last
                start = limit
        out = torch.cat(outs, dim=2)
        return out if self.conv else (out if last else F.gelu(out), None)
def create_sin_embedding(length, dim, shift=0, device="cpu", max_period=10000): pos = shift + torch.arange(length, device=device).view(-1, 1, 1); half = dim // 2; phase = pos / (max_period ** (torch.arange(half, device=device).view(1, 1, -1) / (half - 1))); return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
def create_2d_sin_embedding(d_model, height, width, device="cpu", max_period=10000):
    pe = torch.zeros(1, d_model, height, width, device=device)
    half = d_model // 2
    div = torch.exp(torch.arange(0.0, half, 2, device=device) * -(math.log(max_period) / half))
    pos_w, pos_h = torch.arange(0.0, width, device=device).unsqueeze(1), torch.arange(0.0, height, device=device).unsqueeze(1)
    pe[0, 0:half:2] = torch.sin(pos_w * div).t().unsqueeze(1)
    pe[0, 1:half:2] = torch.cos(pos_w * div).t().unsqueeze(1)
    pe[0, half::2] = torch.sin(pos_h * div).t().unsqueeze(2)
    pe[0, half + 1 :: 2] = torch.cos(pos_h * div).t().unsqueeze(2)
    return pe
def create_sin_embedding_cape(length, dim, batch_size, mean_normalize, augment, max_global_shift=0.0, max_local_shift=0.0, max_scale=1.0, device="cpu", max_period=10000.0):
    pos = torch.arange(length).view(-1, 1, 1).float().repeat(1, batch_size, 1)
    if mean_normalize: pos -= torch.nanmean(pos, dim=0, keepdim=True)
    if augment:  # train-only CAPE jitter: global/local shift + log-scale stretch
        pos = (pos + np.random.uniform(-max_global_shift, max_global_shift, (1, batch_size, 1)) + np.random.uniform(-max_local_shift, max_local_shift, (length, batch_size, 1)))
        pos = pos * np.exp(np.random.uniform(-np.log(max_scale), np.log(max_scale), (1, batch_size, 1)))
    pos, half = pos.to(device), dim // 2
    phase = pos / (max_period ** (torch.arange(half, device=device).view(1, 1, -1) / (half - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1).float()
class MyGroupNorm(nn.GroupNorm):
    def forward(self, x): return super().forward(x.transpose(1, 2)).transpose(1, 2)
class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, group_norm=0, norm_first=False, norm_out=False, layer_norm_eps=1e-5, layer_scale=False, init_values=1e-4, batch_first=False, sparse=False, **kwargs):
        if sparse: raise NotImplementedError("Sparse Demucs transformer is not supported")
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first, device=kwargs.get("device"), dtype=kwargs.get("dtype"))
        if group_norm: self.norm1, self.norm2 = (MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps), MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps))
        self.norm_out = MyGroupNorm(int(norm_out), d_model) if self.norm_first & norm_out else None
        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first: x = x + self.gamma_1(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)); x = x + self.gamma_2(self._ff_block(self.norm2(x))); return self.norm_out(x) if self.norm_out else x
        x = self.norm1(x + self.gamma_1(self._sa_block(x, src_mask, src_key_padding_mask)))
        return self.norm2(x + self.gamma_2(self._ff_block(x)))
class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, layer_scale=False, init_values=1e-4, norm_first=False, group_norm=False, norm_out=False, sparse=False, batch_first=False, **_):
        if sparse: raise NotImplementedError("Sparse Demucs transformer is not supported")
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1, self.dropout, self.linear2 = (nn.Linear(d_model, dim_feedforward), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        norm = ((lambda: MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps)) if group_norm else (lambda: nn.LayerNorm(d_model, eps=layer_norm_eps)))
        self.norm_first = norm_first
        self.norm1, self.norm2, self.norm3 = norm(), norm(), norm()
        self.norm_out = MyGroupNorm(int(norm_out), d_model) if self.norm_first & norm_out else None
        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.dropout1, self.dropout2, self.activation = nn.Dropout(dropout), nn.Dropout(dropout), activation
    def forward(self, q, k, mask=None):
        if self.norm_first:
            norm_q, norm_k = self.norm1(q), self.norm2(k)
            attn_out = self.cross_attn(norm_q, norm_k, norm_k, attn_mask=mask, need_weights=False)[0]
            x = q + self.gamma_1(self.dropout1(attn_out))
            ffn_out = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(x)))))
            x = x + self.gamma_2(self.dropout2(ffn_out))
            return self.norm_out(x) if self.norm_out else x
        attn_out = self.cross_attn(q, k, k, attn_mask=mask, need_weights=False)[0]
        x = self.norm1(q + self.gamma_1(self.dropout1(attn_out)))
        return self.norm2(x + self.gamma_2(self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))))
class PositionEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, scale=1.0, boost=3.0): super().__init__(); self.embedding = nn.Embedding(num_embeddings, embedding_dim); self.embedding.weight.data *= scale / boost; self.boost = boost
    def forward(self, x): return self.embedding(x) * self.boost
class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, emb="sin", hidden_scale=4.0, num_heads=8, num_layers=6, cross_first=False, dropout=0.0,
                 max_positions=1000, norm_in=True, norm_in_group=False, group_norm=False, norm_first=False, norm_out=False,
                 max_period=10000.0, weight_decay=0.0, lr=None, layer_scale=False, gelu=True, sin_random_shift=0,
                 weight_pos_embed=1.0, cape_mean_normalize=True, cape_augment=True, cape_glob_loc_scale=None,
                 sparse_self_attn=False, sparse_cross_attn=False, mask_type="diag", mask_random_seed=42,
                 sparse_attn_window=500, global_window=50, auto_sparsity=False, sparsity=0.95):
        super().__init__()
        self.num_layers, self.classic_parity = num_layers, 1 if cross_first else 0
        self.emb, self.max_period, self.weight_decay, self.lr = emb, max_period, weight_decay, lr
        self.weight_pos_embed, self.sin_random_shift = weight_pos_embed, sin_random_shift
        self.cape_mean_normalize, self.cape_augment = cape_mean_normalize, cape_augment
        self.cape_glob_loc_scale = cape_glob_loc_scale or [5000.0, 1.0, 1.4]
        if emb == "scaled": self.position_embeddings = PositionEmbedding(max_positions, dim, scale=0.2)
        if norm_in:
            norm_fn = lambda: nn.LayerNorm(dim)
        elif norm_in_group: norm_fn = lambda: MyGroupNorm(int(norm_in_group), dim)
        else: norm_fn = lambda: nn.Identity()
        self.norm_in, self.norm_in_t = norm_fn(), norm_fn()
        common = {"d_model": dim, "nhead": num_heads, "dim_feedforward": int(dim * hidden_scale), "dropout": dropout, "activation": F.gelu if gelu else F.relu, "group_norm": group_norm, "norm_first": norm_first, "norm_out": norm_out, "layer_scale": layer_scale, "mask_type": mask_type, "mask_random_seed": mask_random_seed, "sparse_attn_window": sparse_attn_window, "global_window": global_window, "sparsity": sparsity, "auto_sparsity": auto_sparsity, "batch_first": True}
        self.layers, self.layers_t = nn.ModuleList(), nn.ModuleList()
        for idx in range(num_layers): klass = MyTransformerEncoderLayer if idx % 2 == self.classic_parity else CrossTransformerEncoderLayer; sparse = sparse_self_attn if klass is MyTransformerEncoderLayer else sparse_cross_attn; self.layers.append(klass(**common, sparse=sparse)); self.layers_t.append(klass(**common, sparse=sparse))
    def _get_pos_embedding(self, t, b, c, device):
        if self.emb == "sin": return create_sin_embedding(t, c, random.randrange(self.sin_random_shift + 1), device, self.max_period)
        if self.emb == "cape": s = self.cape_glob_loc_scale; return create_sin_embedding_cape(t, c, b, self.cape_mean_normalize, self.training and self.cape_augment, s[0], s[1], s[2], device, self.max_period)
        if self.emb == "scaled": return self.position_embeddings(torch.arange(t, device=device))[:, None]
        raise ValueError(f"unsupported Demucs positional embedding: {self.emb}")
    def forward(self, x, xt):
        b, c, fr, t1 = x.shape
        pos = create_2d_sin_embedding(c, fr, t1, x.device, self.max_period).permute(0, 3, 2, 1).reshape(1, t1 * fr, c)
        x = self.norm_in(x.permute(0, 3, 2, 1).reshape(b, t1 * fr, c)) + self.weight_pos_embed * pos
        b, c, t2 = xt.shape
        xt_pos = self._get_pos_embedding(t2, b, c, x.device).permute(1, 0, 2)
        xt = self.norm_in_t(xt.permute(0, 2, 1)) + self.weight_pos_embed * xt_pos
        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x, xt = self.layers[idx](x), self.layers_t[idx](xt)
            else:
                old_x = x
                x, xt = self.layers[idx](x, xt), self.layers_t[idx](xt, old_x)
        return x.reshape(b, t1, fr, c).permute(0, 3, 2, 1), xt.permute(0, 2, 1)
    def make_optim_group(self):
        group = {"params": list(self.parameters()), "weight_decay": self.weight_decay}
        if self.lr is not None: group['lr'] = self.lr
        return group