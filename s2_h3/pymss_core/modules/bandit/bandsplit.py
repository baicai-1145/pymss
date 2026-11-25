import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
class NormFC(nn.Module):
    def __init__(self, emb_dim, bandwidth, in_channels, normalize_channel_independently=False, treat_channel_as_feature=True):
        super().__init__()
        if normalize_channel_independently: raise NotImplementedError
        self.treat_channel_as_feature = treat_channel_as_feature
        self.norm = nn.LayerNorm(in_channels * bandwidth * 2)
        fc_in = bandwidth * 2 * in_channels if treat_channel_as_feature else bandwidth * 2
        if not treat_channel_as_feature:
            assert emb_dim % in_channels == 0
            emb_dim //= in_channels
        self.fc = nn.Linear(fc_in, emb_dim)
    def forward(self, xb): b, t, c, ribw = xb.shape; xb = self.norm(xb.reshape(b, t, c * ribw)); return self.fc(xb) if (self.treat_channel_as_feature) else self.fc(xb.reshape(b, t, c, ribw)).reshape(b, t, -1)
class SequentialNormFC(nn.Module):
    def __init__(self, emb_dim, bandwidth, in_channels, normalize_channel_independently=False, treat_channel_as_feature=True):
        super().__init__()
        if not treat_channel_as_feature or normalize_channel_independently: raise NotImplementedError
        self.combined = nn.Sequential(nn.LayerNorm(in_channels * bandwidth * 2), nn.Linear(in_channels * bandwidth * 2, emb_dim))
    def forward(self, xb): return checkpoint_sequential(self.combined, 1, xb, use_reentrant=False)
class BandSplitModuleBase(nn.Module):
    def __init__(self, band_specs, emb_dim, in_channels, norm_fc_cls, complex_order, flatten_input, require_no_overlap=False, require_no_gap=True, normalize_channel_independently=False, treat_channel_as_feature=True):
        from .core.model.bsrnn.utils import band_widths_from_specs, check_no_gap, check_no_overlap, check_nonzero_bandwidth
        super().__init__()
        check_nonzero_bandwidth(band_specs)
        if require_no_gap: check_no_gap(band_specs)
        if require_no_overlap: check_no_overlap(band_specs)
        self.band_specs, self.band_widths, self.n_bands = band_specs, band_widths_from_specs(band_specs), len(band_specs)
        self.emb_dim, self.complex_order, self.flatten_input = emb_dim, complex_order, flatten_input
        self.norm_fc_modules = nn.ModuleList([ norm_fc_cls(emb_dim=emb_dim, bandwidth=bw, in_channels=in_channels, normalize_channel_independently=normalize_channel_independently, treat_channel_as_feature=treat_channel_as_feature) for bw in self.band_widths])
    def _band_view(self, x):
        xr = torch.view_as_real(x)
        if self.complex_order == "reim_freq": return xr.permute(0, 3, 1, 4, 2)
        if self.complex_order == "freq_reim": return xr.permute(0, 3, 1, 2, 4).contiguous()
        raise ValueError(f"unsupported complex_order: {self.complex_order}")
    def forward(self, x):
        b, c, _, t = x.shape
        xr = self._band_view(x)
        z = torch.empty(b, self.n_bands, t, self.emb_dim, device=x.device)
        for i, nfm in enumerate(self.norm_fc_modules): f0, f1 = self.band_specs[i]; xb = (xr[..., f0:f1].reshape(b, t, c, -1) if self.complex_order == "reim_freq" else xr[:, :, :, f0:f1].reshape(b, t, -1)); z[:, i] = nfm((xb.reshape(b, t, -1) if self.flatten_input else xb).contiguous())
        return z
class _ConfiguredBandSplitModule(BandSplitModuleBase):
    def __init__(self, band_specs, emb_dim, in_channels, require_no_overlap=False, require_no_gap=True, normalize_channel_independently=False, treat_channel_as_feature=True): super().__init__(band_specs, emb_dim, in_channels, self.norm_fc_cls, self.complex_order, self.flatten_input, require_no_overlap, require_no_gap, normalize_channel_independently, treat_channel_as_feature)