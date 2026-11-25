import torch
from torch import nn
from ..bandit.core.model._spectral import _SpectralComponent
from ..bandit.core.model.bsrnn.utils import MusicalBandsplitSpecification
from ..mlx_backend import MpsBackendMixin
from . import BandSplitModule, OverlappingMaskEstimationModule, SeqBandModellingModule
class Bandit(MpsBackendMixin, _SpectralComponent):
    def __init__(self, in_channels, stems, fs=44100, band_type="musical", n_bands=64, require_no_overlap=False,
                 require_no_gap=True, normalize_channel_independently=False, treat_channel_as_feature=True,
                 n_sqm_modules=12, emb_dim=128, rnn_dim=256, bidirectional=True, rnn_type="LSTM", mlp_dim=512,
                 hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True, use_freq_weights=True,
                 n_fft=2048, win_length=2048, hop_length=512, window_fn="hann_window", wkwargs=None, power=None,
                 center=True, normalized=True, pad_mode="constant", onesided=True):
        super().__init__(n_fft=n_fft, win_length=win_length, hop_length=hop_length, window_fn=window_fn, wkwargs=wkwargs, power=power, normalized=normalized, center=center, pad_mode=pad_mode, onesided=onesided)
        assert band_type == "musical"
        self.in_channels, self.stems = in_channels, stems
        self.band_specs = MusicalBandsplitSpecification(nfft=n_fft, fs=fs, n_bands=n_bands)
        self.band_split = BandSplitModule(in_channels=in_channels, band_specs=self.band_specs.get_band_specs(), require_no_overlap=require_no_overlap, require_no_gap=require_no_gap, normalize_channel_independently=normalize_channel_independently, treat_channel_as_feature=treat_channel_as_feature, emb_dim=emb_dim)
        self.tf_model = SeqBandModellingModule(n_modules=n_sqm_modules, emb_dim=emb_dim, rnn_dim=rnn_dim, bidirectional=bidirectional, rnn_type=rnn_type)
        self.mask_estim = nn.ModuleDict({ stem: OverlappingMaskEstimationModule(band_specs=self.band_specs.get_band_specs(), freq_weights=self.band_specs.get_freq_weights(), n_freq=n_fft // 2 + 1, emb_dim=emb_dim, mlp_dim=mlp_dim, in_channels=in_channels, hidden_activation=hidden_activation, hidden_activation_kwargs=hidden_activation_kwargs or {}, complex_mask=complex_mask, use_freq_weights=use_freq_weights) for stem in stems})
    def mlx_forward_mx(self, raw_audio): from ..bandit_mlx import mlx_forward_bandit_mx; return mlx_forward_bandit_mx(self, raw_audio, self.mps_model_compute_dtype)
    def _use_mlx_full_forward(self, batch): return (not self.training and self.mps_model_backend == "mlx_full" and not isinstance(batch, dict) and batch.device.type == "mps")
    @staticmethod
    def mask(x, m): return x * m
    def forward(self, batch, mode="train"):
        if self._use_mlx_full_forward(batch):
            try:
                from ..bandit_mlx import mlx_forward_bandit
                return mlx_forward_bandit(self, batch, self.mps_model_compute_dtype)
            except Exception as exc:
                self._pymss_mlx_full_backend_error = repr(exc)
                self.mps_model_backend = "torch"
        init_shape = batch.shape
        if not isinstance(batch, dict): batch = {'mixture': {'audio': batch.view(-1, 1, batch.shape[-1])}}
        with torch.no_grad():
            mixture = batch["mixture"]["audio"]
            batch["mixture"]["spectrogram"] = self.stft(mixture)
            for stem in batch.get("sources", {}): batch["sources"][stem]["spectrogram"] = self.stft(batch["sources"][stem]["audio"])
        batch = self.separate(batch)
        return torch.stack([batch["estimates"][s]["audio"].view(-1, init_shape[1], init_shape[2]) for s in self.stems], dim=1)
    def encode(self, batch): x = batch["mixture"]["spectrogram"]; return x, self.tf_model(self.band_split(x)), batch["mixture"]["audio"].shape[-1]
    def separate(self, batch):
        batch["estimates"] = {}
        x, q, length = self.encode(batch)
        for stem, mem in self.mask_estim.items(): s = self.mask(x, mem(q).to(x.dtype)); batch["estimates"][stem] = {"audio": self.istft(s, length), "spectrogram": s}
        return batch