from .bs_roformer import BSRoformer
from .mel_band_roformer import MelBandRoformer
def _conformer_variant(base, inject_norm_output=False):
    # conformer variant: kw names time/freq_conformer_depth + conformer=True + norm_output=False where the base omitted it
    class _Variant(base):
        def __init__(self, dim, *, time_conformer_depth=2, freq_conformer_depth=2, zero_dc=True, **kwargs):
            if inject_norm_output: kwargs['norm_output'] = False
            super().__init__(dim, time_transformer_depth=time_conformer_depth, freq_transformer_depth=freq_conformer_depth, conformer=True, zero_dc=zero_dc, **kwargs)
    return _Variant
BSConformer = _conformer_variant(BSRoformer)
BSConformer.__name__ = BSConformer.__qualname__ = "BSConformer"
MelBandConformer = _conformer_variant(MelBandRoformer, inject_norm_output=True)
MelBandConformer.__name__ = MelBandConformer.__qualname__ = "MelBandConformer"
from .bs_roformer_hyperace import BSRoformerHyperACE
__all__ = ("BSConformer", "BSRoformer", "BSRoformerHyperACE", "MelBandConformer", "MelBandRoformer")