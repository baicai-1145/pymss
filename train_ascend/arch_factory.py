"""Single place that decides which band U-Net implementation to build."""

from __future__ import annotations

from torch.nn import Module


def build(arch: str = "v1", n_fft: int = 2048) -> Module:
    if arch in ("v1", "freq_unet"):
        from freq_unet import FreqUNetConfig, build_model

        return build_model(FreqUNetConfig(n_fft=n_fft))
    if arch in ("v3", "freq_unet_v3"):
        from freq_unet_v3 import FreqUNetV3Config, build_model

        return build_model(FreqUNetV3Config(n_fft=n_fft))
    raise SystemExit(f"unknown arch {arch!r} (expected v1 or v3)")


def param_count(model: Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
