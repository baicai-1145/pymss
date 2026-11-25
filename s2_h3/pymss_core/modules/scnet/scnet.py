import math
import torch
import torch.nn.functional as F
from torch import nn
from ..mlx_backend import MpsBackendMixin
from pymss_core.npu_fft import istft_npu, stft_npu, torch_stft_blocked
from .separation import SeparationNet
class Swish(nn.Module):
    def forward(self, x): return x * x.sigmoid()
class ConvolutionModule(nn.Module):
    def __init__(self, channels, depth=2, compress=4, kernel=3):
        super().__init__()
        assert kernel % 2 == 1
        h = int(channels / compress)
        self.layers = nn.ModuleList([nn.Sequential(nn.GroupNorm(1, channels), nn.Conv1d(channels, h * 2, kernel, padding=kernel // 2), nn.GLU(1), nn.Conv1d(h, h, kernel, padding=kernel // 2, groups=h), nn.GroupNorm(1, h), Swish(), nn.Conv1d(h, channels, 1)) for _ in range(abs(depth))])
    def forward(self, x):
        for layer in self.layers: x = x + layer(x)
        return x
class FusionLayer(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1): super().__init__(); self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=padding)
    def forward(self, x, skip=None):
        if skip is not None: x = x + skip
        return F.glu(self.conv(x.repeat(1, 2, 1, 1)), dim=1)
class SDlayer(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs): super().__init__(); self.convs = nn.ModuleList([nn.Conv2d(channels_in, channels_out, (c["kernel"], 1), (c["stride"], 1)) for c in band_configs.values()]); self.strides = [c["stride"] for c in band_configs.values()]; self.kernels = [c["kernel"] for c in band_configs.values()]; self.SR_low, self.SR_mid = band_configs["low"]["SR"], band_configs["mid"]["SR"]
    def forward(self, x):
        Fr = x.shape[2]
        low, mid = math.ceil(Fr * self.SR_low), math.ceil(Fr * (self.SR_low + self.SR_mid))
        outputs, original_lengths = [], []
        for conv, stride, kernel, (s, e) in zip(self.convs, self.strides, self.kernels, [(0, low), (low, mid), (mid, Fr)]): p = kernel - stride if stride == 1 else (stride - (e - s) % stride) % stride; outputs.append(conv(F.pad(x[:, :, s:e], (0, 0, p // 2, p - p // 2)))); original_lengths.append(e - s)
        return outputs, original_lengths
class SUlayer(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs): super().__init__(); self.convtrs = nn.ModuleList([nn.ConvTranspose2d(channels_in, channels_out, [c["kernel"], 1], [c["stride"], 1]) for c in band_configs.values()])
    def forward(self, x, lengths, origin_lengths):
        outs = []
        for idx, (convtr, (s, e)) in enumerate(zip(self.convtrs, [(0, lengths[0]), (lengths[0], lengths[0] + lengths[1]), (lengths[0] + lengths[1], None)])): out = convtr(x[:, :, s:e]); dist = abs(origin_lengths[idx] - out.shape[2]) // 2; outs.append(out[:, :, dist:dist + origin_lengths[idx]])
        return torch.cat(outs, dim=2)
class SDblock(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs=None, conv_config=None, depths=None, kernel_size=3):
        if depths is None: depths = [3, 2, 1]
        if conv_config is None: conv_config = {}
        if band_configs is None: band_configs = {}
        super().__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)
        self.conv_modules = nn.ModuleList([ConvolutionModule(channels_out, depth, **conv_config) for depth in depths])
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, 1, (kernel_size - 1) // 2)
    def forward(self, x): bands, original_lengths = self.SDlayer(x); bands = [F.gelu(conv(band.permute(0, 2, 1, 3).reshape(-1, band.shape[1], band.shape[3])) .view(band.shape[0], band.shape[2], band.shape[1], band.shape[3]).permute(0, 2, 1, 3)) for conv, band in zip(self.conv_modules, bands)]; full_band = torch.cat(bands, dim=2); return self.globalconv(full_band), full_band, [b.size(-2) for b in bands], original_lengths
class SCNet(MpsBackendMixin, nn.Module):
    def __init__(self, sources=None, audio_channels=2, dims=None, nfft=4096, hop_size=1024, win_size=4096, normalized=True, band_SR=None, band_stride=None, band_kernel=None, conv_depths=None, compress=4, conv_kernel=3, num_dplayer=6, expand=1):
        if conv_depths is None: conv_depths = [3, 2, 1]
        if band_kernel is None: band_kernel = [3, 4, 16]
        if band_stride is None: band_stride = [1, 4, 16]
        if band_SR is None: band_SR = [0.175, 0.392, 0.433]
        if dims is None: dims = [4, 32, 64, 128]
        if sources is None: sources = ['drums', 'bass', 'other', 'vocals']
        super().__init__()
        self.sources, self.audio_channels, self.dims = sources, audio_channels, dims
        self.band_configs = {k: {"SR": sr, "stride": st, "kernel": k2} for k, sr, st, k2 in zip(["low", "mid", "high"], band_SR, band_stride, band_kernel)}
        self.hop_length = hop_size
        self.conv_config = {"compress": compress, "kernel": conv_kernel}
        self.stft_config = {"n_fft": nfft, "hop_length": hop_size, "win_length": win_size, "center": True, "normalized": normalized}
        self.encoder = nn.ModuleList([SDblock(dims[i], dims[i + 1], self.band_configs, self.conv_config, conv_depths) for i in range(len(dims) - 1)])
        self.decoder = nn.ModuleList([nn.Sequential(FusionLayer(dims[i + 1]), SUlayer(dims[i + 1], dims[i] if i else dims[i] * len(sources), self.band_configs)) for i in reversed(range(len(dims) - 1))])
        self.separation_net = SeparationNet(channels=dims[-1], expand=expand, num_layers=num_dplayer)
    def mlx_forward_mx(self, raw_audio): from ..scnet_mlx import mlx_forward_scnet_mx; return mlx_forward_scnet_mx(self, raw_audio, self.mps_model_compute_dtype)
    def forward(self, x):
        if self._use_mlx_full_forward(x):
            try:
                from ..scnet_mlx import mlx_forward_scnet
                return mlx_forward_scnet(self, x, self.mps_model_compute_dtype)
            except Exception as exc:
                self._pymss_mlx_full_backend_error = repr(exc)
                self.mps_model_backend = "torch"
        B = x.shape[0]
        padding = self.hop_length - x.shape[-1] % self.hop_length
        if (x.shape[-1] + padding) // self.hop_length % 2 == 0: padding += self.hop_length
        x = F.pad(x, (0, padding))
        fft_fallback = torch_stft_blocked(x.device)
        stft_fn = stft_npu if fft_fallback else torch.stft
        istft_fn = istft_npu if fft_fallback else torch.istft
        x = torch.view_as_real(stft_fn(x.reshape(-1, x.shape[-1]), **self.stft_config, return_complex=True))
        x = x.permute(0, 3, 1, 2).reshape(B, self.audio_channels * 2, x.shape[1], x.shape[2])
        _B, _C, Fr, T = x.shape
        saved = []
        for sd_layer in self.encoder: x, skip, lengths, original_lengths = sd_layer(x); saved.append((skip, lengths, original_lengths))
        x = self.separation_net(x)
        for fusion_layer, su_layer in self.decoder: skip, lengths, original_lengths = saved.pop(); x = su_layer(fusion_layer(x, skip), lengths, original_lengths)
        x = istft_fn(torch.view_as_complex(x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1).contiguous()), **self.stft_config)
        return x.reshape(B, len(self.sources), self.audio_channels, -1)[:, :, :, :-padding]