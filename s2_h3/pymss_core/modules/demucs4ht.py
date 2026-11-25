import math
from fractions import Fraction
import torch
from torch import nn
from torch.nn import functional as F
from ..config import to_plain
from .demucs_local import (CrossTransformerEncoder, HDecLayer, HEncLayer, MultiWrap, ScaledEmbedding, ispectro, pad1d, rescale_module, spectro)
from .mlx_backend import MpsBackendMixin
class HTDemucs(MpsBackendMixin, nn.Module):
    def __init__(
        self,
        sources, audio_channels=2, channels=48, channels_time=None, growth=2, nfft=4096, num_subbands=1,
        wiener_iters=0, end_iters=0, wiener_residual=False, cac=True, depth=4, rewrite=True, multi_freqs=None,
        multi_freqs_depth=3, freq_emb=0.2, emb_scale=10, emb_smooth=True, kernel_size=8, time_stride=2, stride=4,
        context=1, context_enc=0, norm_starts=4, norm_groups=4, dconv_mode=1, dconv_depth=2, dconv_comp=8,
        dconv_init=1e-3, bottom_channels=0, t_layers=5, t_emb="sin", t_hidden_scale=4.0, t_heads=8, t_dropout=0.0,
        t_max_positions=10000, t_norm_in=True, t_norm_in_group=False, t_group_norm=False, t_norm_first=True,
        t_norm_out=True, t_max_period=10000.0, t_weight_decay=0.0, t_lr=None, t_layer_scale=True, t_gelu=True,
        t_weight_pos_embed=1.0, t_sin_random_shift=0, t_cape_mean_normalize=True, t_cape_augment=True,
        t_cape_glob_loc_scale=None, t_sparse_self_attn=False, t_sparse_cross_attn=False,
        t_mask_type="diag", t_mask_random_seed=42, t_sparse_attn_window=500, t_global_window=100, t_sparsity=0.95,
        t_auto_sparsity=False, t_cross_first=False, rescale=0.1, samplerate=44100, segment=10, use_train_segment=False,
    ):
        if t_cape_glob_loc_scale is None: t_cape_glob_loc_scale = [5000.0, 1.0, 1.4]
        super().__init__()
        self.num_subbands, self.cac, self.wiener_residual, self.audio_channels, self.sources = (num_subbands, cac, wiener_residual, audio_channels, sources)
        self.kernel_size, self.context, self.stride, self.depth, self.bottom_channels = kernel_size, context, stride, depth, bottom_channels
        self.channels, self.samplerate, self.segment, self.use_train_segment = channels, samplerate, segment, use_train_segment
        self.nfft, self.hop_length, self.wiener_iters, self.end_iters, self.freq_emb = nfft, nfft // 4, wiener_iters, end_iters, None
        assert wiener_iters == end_iters
        self.encoder, self.decoder, self.tencoder, self.tdecoder = (nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList())
        chin = audio_channels
        zmul = (2 if cac else 1) * (max(1, num_subbands))  # freq branch channels: CaC x subbands
        chin_z, freqs = chin * zmul, nfft // 2
        chout, chout_z = channels_time or channels, channels
        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            if not freq:
                assert freqs == 1
                ker, stri, pad, last_freq = time_stride * 2, time_stride, True, False
            elif freqs <= kernel_size: ker, stri, pad, last_freq = (freqs, stride, False, True)
            else: ker, stri, pad, last_freq = (kernel_size, stride, True, False)
            kw = { "kernel_size": ker, "stride": stri, "freq": freq, "pad": pad, "norm": norm, "rewrite": rewrite, "norm_groups": norm_groups, "dconv_kw": {"depth": dconv_depth, "compress": dconv_comp, "init": dconv_init, "gelu": True}, }
            kwt = dict(kw, freq=0, kernel_size=kernel_size, stride=stride, pad=True)  # time branch
            kw_dec = dict(kw)
            multi = bool(multi_freqs and index < multi_freqs_depth)
            if multi: kw_dec['context_freq'] = False
            if last_freq: chout_z = max(chout, chout_z); chout = chout_z
            enc = HEncLayer(chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw)
            if freq: self.tencoder.append(HEncLayer(chin, chout, dconv=dconv_mode & 1, context=context_enc, empty=last_freq, **kwt))
            if multi: enc = MultiWrap(enc, multi_freqs)
            self.encoder.append(enc)
            if index == 0: chin = self.audio_channels * len(self.sources); chin_z = chin * zmul
            dec = HDecLayer(chout_z, chin_z, dconv=dconv_mode & 2, last=index == 0, context=context, **kw_dec)
            if multi: dec = MultiWrap(dec, multi_freqs)
            if freq: self.tdecoder.insert(0, HDecLayer(chout, chin, dconv=dconv_mode & 2, empty=last_freq, last=index == 0, context=context, **kwt))
            self.decoder.insert(0, dec)
            chin, chin_z, chout, chout_z = chout, chout_z, int(growth * chout), int(growth * chout_z)
            if freq: freqs = 1 if freqs <= kernel_size else freqs // stride
            if index == 0 and freq_emb: self.freq_emb = ScaledEmbedding(freqs, chin_z, smooth=emb_smooth, scale=emb_scale); self.freq_emb_scale = freq_emb
        if rescale: rescale_module(self, reference=rescale)
        transformer_channels = channels * growth ** (depth - 1)
        if bottom_channels: self.channel_upsampler, self.channel_downsampler = (nn.Conv1d(transformer_channels, bottom_channels, 1), nn.Conv1d(bottom_channels, transformer_channels, 1)); self.channel_upsampler_t, self.channel_downsampler_t = (nn.Conv1d(transformer_channels, bottom_channels, 1), nn.Conv1d(bottom_channels, transformer_channels, 1)); transformer_channels = bottom_channels
        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels, emb=t_emb, hidden_scale=t_hidden_scale, num_heads=t_heads, num_layers=t_layers,
                cross_first=t_cross_first, dropout=t_dropout, max_positions=t_max_positions, norm_in=t_norm_in,
                norm_in_group=t_norm_in_group, group_norm=t_group_norm, norm_first=t_norm_first, norm_out=t_norm_out,
                max_period=t_max_period, weight_decay=t_weight_decay, lr=t_lr, layer_scale=t_layer_scale, gelu=t_gelu,
                sin_random_shift=t_sin_random_shift, weight_pos_embed=t_weight_pos_embed,
                cape_mean_normalize=t_cape_mean_normalize, cape_augment=t_cape_augment,
                cape_glob_loc_scale=t_cape_glob_loc_scale, sparse_self_attn=t_sparse_self_attn,
                sparse_cross_attn=t_sparse_cross_attn, mask_type=t_mask_type, mask_random_seed=t_mask_random_seed,
                sparse_attn_window=t_sparse_attn_window, global_window=t_global_window, sparsity=t_sparsity,
                auto_sparsity=t_auto_sparsity)
        else:
            self.crosstransformer = None
    def mlx_forward_mx(self, raw_audio): from .demucs_mlx import mlx_forward_demucs_mx; return mlx_forward_demucs_mx(self, raw_audio, self.mps_model_compute_dtype)
    def _spec(self, x):
        hl, nfft = self.hop_length, self.nfft
        assert hl == nfft // 4
        le = math.ceil(x.shape[-1] / hl)
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        return z[..., 2 : 2 + le]
    def _ispec(self, z, length=None, scale=0): hl = self.hop_length // (4**scale); z = F.pad(z, (2, 2, 0, 1)); pad = hl // 2 * 3; le = hl * math.ceil(length / hl) + 2 * pad; return ispectro(z, hl, length=le)[..., pad : pad + length]
    def _magnitude(self, z):
        if self.cac: B, C, Fr, T = z.shape; return torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(B, C * 2, Fr, T)
        return z.abs()
    def _mask(self, z, m):
        niters = self.end_iters if self.training else self.wiener_iters
        if self.cac: B, S, _C, Fr, T = m.shape; return torch.view_as_complex(m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3).contiguous())
        if niters < 0: z = z[:, None]; return z / (1e-08 + z.abs()) * m
        return self._wiener(m, z, niters)
    def _wiener(self, mag_out, mix_stft, niters): raise NotImplementedError("non-CaC Wiener Demucs is not supported by the dependency-free path")
    def valid_length(self, length):
        if not self.use_train_segment: return length
        training_length = int(self.segment * self.samplerate)
        if training_length < length: raise ValueError(f"Given length {length} is longer than training length {training_length}")
        return training_length
    def cac2cws(self, x): b, c, f, t = x.shape; return x.reshape(b, c * self.num_subbands, f // self.num_subbands, t)
    def cws2cac(self, x): b, c, f, t = x.shape; return x.reshape(b, c // self.num_subbands, f * self.num_subbands, t)
    def forward(self, mix):
        if self._use_mlx_full_forward(mix):
            try:
                from .demucs_mlx import mlx_forward_demucs
                return mlx_forward_demucs(self, mix, self.mps_model_compute_dtype)
            except Exception as exc:
                self._pymss_mlx_full_backend_error = repr(exc)
                self.mps_model_backend = "torch"
        length = mix.shape[-1]
        length_pre_pad = None
        if self.use_train_segment:
            if self.training:
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                training_length = int(self.segment * self.samplerate)
                if mix.shape[-1] < training_length: length_pre_pad = mix.shape[-1]; mix = F.pad(mix, (0, training_length - length_pre_pad))
        z = self._spec(mix)
        x = self._magnitude(z) if self.num_subbands <= 1 else self.cac2cws(self._magnitude(z))
        B, _, Fq, T = x.shape
        mean, std = x.mean(dim=(1, 2, 3), keepdim=True), x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        meant, stdt = mix.mean(dim=(1, 2), keepdim=True), mix.std(dim=(1, 2), keepdim=True)
        xt = (mix - meant) / (1e-5 + stdt)
        saved, saved_t, lengths_t = [], [], []
        for idx, encode in enumerate(self.encoder):
            skip_length = x.shape[-1]
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if tenc.empty: inject = xt
                else: saved_t.append(xt)
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = (self.freq_emb_scale * self.freq_emb(frs).t())[None, :, :, None].expand_as(x)
                try:
                    x = x + emb.contiguous()
                except RuntimeError:
                    # broadcast pointwise can hit invalid-parameter launches on
                    # the vendor stack; same-shape CPU add is exact
                    x = (x.cpu() + emb.contiguous().cpu()).to(x.device)
            saved.append((x, skip_length))
        if self.crosstransformer:
            if self.bottom_channels: b, c, f, t = x.shape; x = self.channel_upsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t); xt = self.channel_upsampler_t(xt)
            x, xt = self.crosstransformer(x, xt)
            if self.bottom_channels: b, c, f, t = x.shape; x = self.channel_downsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t); xt = self.channel_downsampler_t(xt)
        for idx, decode in enumerate(self.decoder):
            skip, skip_length = saved.pop(-1)
            x, pre = decode(x, skip, skip_length)
            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    xt, _ = tdec(pre[:, :, 0], None, length_t)
                else:
                    xt, _ = tdec(xt, saved_t.pop(-1), length_t)
        assert len(saved) == len(lengths_t) == len(saved_t) == 0
        S = len(self.sources)
        if self.num_subbands > 1: x = self.cws2cac(x.view(B, -1, Fq, T))
        x = x.view(B, S, -1, Fq * self.num_subbands, T) * std[:, None] + mean[:, None]
        zout = self._mask(z, x)
        x = self._ispec(zout, length if not self.use_train_segment or self.training else training_length)
        xt = xt.view(B, S, -1, length if not self.use_train_segment or self.training else training_length)
        x = xt * stdt[:, None] + meant[:, None] + x
        if length_pre_pad: x = x[..., :length_pre_pad]
        return x
def get_model(args):
    extra = { "sources": list(args.training.instruments), "audio_channels": args.training.channels, "samplerate": args.training.samplerate, "segment": args.training.segment, }
    if args.model != "htdemucs": raise ValueError(f"Only htdemucs configs are supported, got {args.model!r}")
    return HTDemucs(**extra, **to_plain(getattr(args, args.model)))