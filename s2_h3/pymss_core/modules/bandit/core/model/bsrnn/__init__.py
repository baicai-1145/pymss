from ....bandsplit import NormFC as _NormFC
from ....bandsplit import _ConfiguredBandSplitModule
from ....maskestim import (BaseNormMLP, MaskEstimationModule, MaskEstimationModuleBase, MaskEstimationModuleSuperBase, MultAddNormMLP, NormMLP, OverlappingMaskEstimationModule)
from ....tfmodel import ResidualRNN, TimeFrequencyModellingModule, Transpose, _SeqBandModellingPreset
class BandSplitModule(_ConfiguredBandSplitModule):
    norm_fc_cls, complex_order, flatten_input = _NormFC, "reim_freq", False
    def __init__(self, band_specs, emb_dim, in_channels=None, in_channel=None, require_no_overlap=False, require_no_gap=True, normalize_channel_independently=False, treat_channel_as_feature=True):
        in_channels = in_channels if in_channels is not None else in_channel
        super().__init__(band_specs=band_specs, emb_dim=emb_dim, in_channels=in_channels, require_no_overlap=require_no_overlap, require_no_gap=require_no_gap, normalize_channel_independently=normalize_channel_independently, treat_channel_as_feature=treat_channel_as_feature)
class SeqBandModellingModule(_SeqBandModellingPreset):
    pass