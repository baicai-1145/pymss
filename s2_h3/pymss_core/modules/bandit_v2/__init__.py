from ..bandit.bandsplit import SequentialNormFC as NormFC
from ..bandit.bandsplit import _ConfiguredBandSplitModule
from ..bandit.core.model.bsrnn.utils import (BandsplitSpecification, BassBandsplitSpecification, DrumBandsplitSpecification, MelBandsplitSpecification, MusicalBandsplitSpecification, OtherBandsplitSpecification, PerceptualBandsplitSpecification, VocalBandsplitSpecification, band_widths_from_specs, check_no_gap, check_no_overlap, check_nonzero_bandwidth, mel_filterbank, musical_filterbank)
from ..bandit.maskestim import BaseNormMLP, MaskEstimationModuleBase, MaskEstimationModuleSuperBase
from ..bandit.maskestim import MaskEstimationModule as _MaskEstimationModule
from ..bandit.maskestim import NormMLP as _NormMLP
from ..bandit.maskestim import OverlappingMaskEstimationModule as _OverlappingMaskEstimationModule
from ..bandit.tfmodel import ResidualRNN, TimeFrequencyModellingModule, Transpose, _SeqBandModellingPreset
class NormMLP(_NormMLP):
    def __init__(self, emb_dim, mlp_dim, bandwidth, in_channels, hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True): super().__init__(emb_dim=emb_dim, mlp_dim=mlp_dim, bandwidth=bandwidth, in_channels=in_channels, hidden_activation=hidden_activation, hidden_activation_kwargs=hidden_activation_kwargs, complex_mask=complex_mask, use_combined=True, use_checkpoint=True)
class OverlappingMaskEstimationModule(_OverlappingMaskEstimationModule):
    def __init__(self, in_channels, band_specs, freq_weights, n_freq, emb_dim, mlp_dim, cond_dim=0, hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True, norm_mlp_cls=NormMLP, norm_mlp_kwargs=None, use_freq_weights=False):
        super().__init__(in_channels=in_channels, band_specs=band_specs, freq_weights=freq_weights, n_freq=n_freq, emb_dim=emb_dim, mlp_dim=mlp_dim, cond_dim=cond_dim, hidden_activation=hidden_activation, hidden_activation_kwargs=hidden_activation_kwargs, complex_mask=complex_mask, norm_mlp_cls=norm_mlp_cls, norm_mlp_kwargs=norm_mlp_kwargs, use_freq_weights=use_freq_weights, register_all_freq_weights=False, allow_cond=False, output_dtype="complex64", compute_all_masks=False)
class MaskEstimationModule(_MaskEstimationModule):
    def __init__(self, band_specs, emb_dim, mlp_dim, in_channels, hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True, **kwargs): super().__init__(band_specs=band_specs, emb_dim=emb_dim, mlp_dim=mlp_dim, in_channels=in_channels, hidden_activation=hidden_activation, hidden_activation_kwargs=hidden_activation_kwargs, complex_mask=complex_mask)
class BandSplitModule(_ConfiguredBandSplitModule):
    norm_fc_cls, complex_order, flatten_input = NormFC, "freq_reim", True
class SeqBandModellingModule(_SeqBandModellingPreset):
    @staticmethod
    def _preset_runtime_options(n_modules, parallel_mode): return {"sequential_transpose": not parallel_mode, "checkpoint_segments": None if parallel_mode else n_modules}
from .bandit import Bandit
__all__ = ("BandSplitModule", "Bandit", "MaskEstimationModule", "NormFC", "NormMLP", "OverlappingMaskEstimationModule", "SeqBandModellingModule")