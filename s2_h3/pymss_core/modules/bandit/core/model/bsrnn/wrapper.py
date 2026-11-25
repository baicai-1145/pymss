import torch
from .....mlx_backend import MpsBackendMixin
from .._spectral import _SpectralComponent
from .core import MultiSourceMultiMaskBandSplitCoreRNN
from .utils import (BarkBandsplitSpecification, EquivalentRectangularBandsplitSpecification, MelBandsplitSpecification, MusicalBandsplitSpecification, TriangularBarkBandsplitSpecification, VocalBandsplitSpecification)
def get_band_specs(band_specs, n_fft, fs, n_bands=None):
    if not isinstance(band_specs, str): return band_specs, None, False
    if band_specs in ("dnr:speech", "dnr:vox7", "musdb:vocals", "musdb:vox7"): return VocalBandsplitSpecification(nfft=n_fft, fs=fs).get_band_specs(), None, False
    for key, cls in (("tribark", TriangularBarkBandsplitSpecification), ("bark", BarkBandsplitSpecification), ("erb", EquivalentRectangularBandsplitSpecification), ("musical", MusicalBandsplitSpecification), ("mel", MelBandsplitSpecification)):
        if key in band_specs:
            assert n_bands is not None
            specs = cls(nfft=n_fft, fs=fs, n_bands=n_bands)
            return specs.get_band_specs(), specs.get_freq_weights(), True
    raise ValueError(f"Unsupported band_specs: {band_specs}")
class MultiMaskMultiSourceBandSplitBaseSimple(MpsBackendMixin, _SpectralComponent):
    def __init__(self, stems, band_specs, fs=44100, n_fft=2048, win_length=2048, hop_length=512, window_fn="hann_window", wkwargs=None, power=None, center=True, normalized=True, pad_mode="constant", onesided=True, n_bands=None):
        super().__init__(n_fft=n_fft, win_length=win_length, hop_length=hop_length, window_fn=window_fn, wkwargs=wkwargs, power=power, center=center, normalized=normalized, pad_mode=pad_mode, onesided=onesided)
        self.band_specs, self.freq_weights, self.overlapping_band = get_band_specs(band_specs, n_fft, fs, n_bands)
        self.stems = stems
    def mlx_forward_mx(self, raw_audio): from .....bandit_mlx import mlx_forward_bandit_mx; return mlx_forward_bandit_mx(self, raw_audio, self.mps_model_compute_dtype)
    def forward(self, batch):
        if self._use_mlx_full_forward(batch):
            try:
                from .....bandit_mlx import mlx_forward_bandit
                return mlx_forward_bandit(self, batch, self.mps_model_compute_dtype)
            except Exception as exc:
                self._pymss_mlx_full_backend_error = repr(exc)
                self.mps_model_backend = "torch"
        with torch.no_grad():
            x = self.stft(batch)
        estimates = [self.istft(spec, batch.shape[-1]) for spec in self.bsrnn(x, cond=None)["spectrogram"].values()]
        return torch.stack(estimates, dim=1)
class MultiMaskMultiSourceBandSplitRNNSimple(MultiMaskMultiSourceBandSplitBaseSimple):
    def __init__(self, in_channel, stems, band_specs, fs=44100, require_no_overlap=False, require_no_gap=True,
                 normalize_channel_independently=False, treat_channel_as_feature=True, n_sqm_modules=12, emb_dim=128,
                 rnn_dim=256, cond_dim=0, bidirectional=True, rnn_type="LSTM", mlp_dim=512, hidden_activation="Tanh",
                 hidden_activation_kwargs=None, complex_mask=True, n_fft=2048, win_length=2048, hop_length=512,
                 window_fn="hann_window", wkwargs=None, power=None, center=True, normalized=True, pad_mode="constant",
                 onesided=True, n_bands=None, use_freq_weights=True, normalize_input=False, mult_add_mask=False,
                 freeze_encoder=False):
        super().__init__(stems=stems, band_specs=band_specs, fs=fs, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window_fn=window_fn, wkwargs=wkwargs, power=power, center=center, normalized=normalized, pad_mode=pad_mode, onesided=onesided, n_bands=n_bands)
        self.bsrnn = MultiSourceMultiMaskBandSplitCoreRNN(
            in_channel=in_channel, stems=stems, band_specs=self.band_specs, require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap, normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature, n_sqm_modules=n_sqm_modules, emb_dim=emb_dim,
            rnn_dim=rnn_dim, bidirectional=bidirectional, rnn_type=rnn_type, mlp_dim=mlp_dim, cond_dim=cond_dim,
            hidden_activation=hidden_activation, hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask, overlapping_band=self.overlapping_band, freq_weights=self.freq_weights,
            n_freq=n_fft // 2 + 1, use_freq_weights=use_freq_weights, mult_add_mask=mult_add_mask)
        self.normalize_input, self.cond_dim = normalize_input, cond_dim
        if freeze_encoder:
            for module in (self.bsrnn.band_split, self.bsrnn.tf_model):
                for param in module.parameters(): param.requires_grad = False