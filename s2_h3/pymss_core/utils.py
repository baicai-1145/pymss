from yaml import YAMLError

from .config import load_config
from .model_detection import ModelTypeDetectionError, detect_model_type
def get_model_from_config(model_type, config_path, model_kwargs_override=None):
    """Instantiate a separation model from a model configuration file."""
    import importlib
    try:
        config = load_config(config_path)
    except (OSError, UnicodeError, YAMLError) as exc:
        if model_type != "auto": raise
        raise ModelTypeDetectionError("Cannot determine model_type: auto requires a readable YAML configuration. Check config_path or set model_type explicitly.") from exc
    if model_type == "auto": model_type = detect_model_type(config)
    if model_type == "mdx23c":
        from .modules.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        return TFC_TDF_net(config), config
    if model_type == "htdemucs":
        from .modules.demucs4ht import get_model
        return get_model(config), config
    if model_type == "vr": raise ValueError("VR network modules do not use YAML config loading")
    models = {
        "mel_band_roformer": ("bs_roformer", "MelBandRoformer", "model"),
        "mel_band_conformer": ("bs_roformer", "MelBandConformer", "model"),
        "bs_roformer": ("bs_roformer", "BSRoformer", "model"),
        "bs_conformer": ("bs_roformer", "BSConformer", "model"),
        "bs_roformer_hyperace": ("bs_roformer", "BSRoformerHyperACE", "model"),
        "bandit": ("bandit.core.model", "MultiMaskMultiSourceBandSplitRNNSimple", "model"),
        "bandit_v2": ("bandit_v2.bandit", "Bandit", "kwargs"),
        "scnet": ("scnet", "SCNet", "model"),
        "apollo": ("look2hear.apollo", "Apollo", "model"),
    }
    if model_type not in models: raise ValueError(f"Model type {model_type} not supported")
    package, class_name, config_key = models[model_type]
    cls = getattr(importlib.import_module(f".modules.{package}", __package__), class_name)
    model_kwargs = dict(config[config_key])
    if config_key == "model":
        # Architecture labels are configuration metadata, not constructor arguments.
        for key in ("type", "model_type", "architecture"): model_kwargs.pop(key, None)
    model_kwargs.update(model_kwargs_override or {})
    return cls(**model_kwargs), config
