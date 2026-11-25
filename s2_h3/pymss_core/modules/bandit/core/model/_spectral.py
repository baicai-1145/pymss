import torch
from torch import nn
from pymss_core.npu_fft import istft_npu, stft_npu, torch_stft_blocked
class _TorchSpectrogram(nn.Module):
    def __init__(self, n_fft, win_length, hop_length, window_fn, wkwargs, normalized, center, pad_mode, onesided): super().__init__(); self.n_fft, self.hop_length, self.normalized = n_fft, hop_length, normalized; self.win_length, self.center, self.pad_mode, self.onesided = win_length or n_fft, center, pad_mode, onesided; self.register_buffer("window", window_fn(self.win_length, **(wkwargs or {})))
    def _spectral_kwargs(self, pad_mode=True):
        kwargs = dict(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=self.center, normalized=self.normalized, onesided=self.onesided)
        if pad_mode: kwargs["pad_mode"] = self.pad_mode
        return kwargs
    def forward(self, x):
        fft_fallback = torch_stft_blocked(x.device)
        fn = stft_npu if fft_fallback else torch.stft
        flat = x.reshape(-1, x.shape[-1])
        if fft_fallback: flat = flat.float()
        spec = fn(flat, **self._spectral_kwargs(), return_complex=True)
        return spec.reshape(*x.shape[:-1], *spec.shape[-2:])
class _TorchInverseSpectrogram(_TorchSpectrogram):
    def forward(self, x, length=None):
        fn = istft_npu if torch_stft_blocked(x.device) else torch.istft
        audio = fn(x.reshape(-1, *x.shape[-2:]), **self._spectral_kwargs(pad_mode=False), length=length, return_complex=False)
        return audio.reshape(*x.shape[:-2], audio.shape[-1])
class _SpectralComponent(nn.Module):
    def __init__(self, n_fft=2048, win_length=2048, hop_length=512, window_fn="hann_window", wkwargs=None, power=None, center=True, normalized=True, pad_mode="constant", onesided=True, **kwargs):
        super().__init__()
        assert power is None
        kwargs = {"n_fft": n_fft, "win_length": win_length, "hop_length": hop_length, "window_fn": torch.__dict__[window_fn], "wkwargs": wkwargs, "normalized": normalized, "center": center, "pad_mode": pad_mode, "onesided": onesided}
        self.stft, self.istft = _TorchSpectrogram(**kwargs), _TorchInverseSpectrogram(**kwargs)
