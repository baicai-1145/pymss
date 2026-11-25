import torch
from torch import nn
from .mlx_backend import MpsBackendMixin
from .spectrogram import SubbandSTFT, forward_subband_mask_model, get_activation
def get_norm(norm_type):
    if norm_type == "BatchNorm": return nn.BatchNorm2d
    if norm_type == "InstanceNorm": return lambda c: nn.InstanceNorm2d(c, affine=True)
    if "GroupNorm" in norm_type: return lambda c: nn.GroupNorm(int(norm_type.replace("GroupNorm", "")), c)
    return lambda c: nn.Identity()
def _block(**modules):
    block = nn.Module(); [block.add_module(n, m) for n, m in modules.items()]; return block
class Upscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__(); self.conv = nn.Sequential(norm(in_c), act, nn.ConvTranspose2d(in_c, out_c, scale, scale, bias=False))
    def forward(self, x): return self.conv(x)
class Downscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__(); self.conv = nn.Sequential(norm(in_c), act, nn.Conv2d(in_c, out_c, scale, scale, bias=False))
    def forward(self, x): return self.conv(x)
class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn, norm, act):
        super().__init__()
        def block():
            nonlocal in_c
            out = _block(tfc1=nn.Sequential(norm(in_c), act, nn.Conv2d(in_c, c, 3, 1, 1, bias=False)), tdf=nn.Sequential(norm(c), act, nn.Linear(f, f // bn, bias=False), norm(c), act, nn.Linear(f // bn, f, bias=False)), tfc2=nn.Sequential(norm(c), act, nn.Conv2d(c, c, 3, 1, 1, bias=False)), shortcut=nn.Conv2d(in_c, c, 1, 1, 0, bias=False))
            in_c = c
            return out
        self.blocks = nn.ModuleList([block() for _ in range(l)])
    def forward(self, x):
        for block in self.blocks: s = block.shortcut(x); x = block.tfc2((x := block.tfc1(x)) + block.tdf(x)) + s
        return x
class TFC_TDF_net(MpsBackendMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        norm, act = get_norm(norm_type=config.model.norm), get_activation(config.model.act)
        self.num_target_instruments = 1 if config.training.target_instrument else len(config.training.instruments)
        self.num_subbands = config.model.num_subbands
        dim_c, m = self.num_subbands * config.audio.num_channels * 2, config.model
        n, scale, l, c, g, bn = m.num_scales, m.scale, m.num_blocks_per_scale, m.num_channels, m.growth, m.bottleneck_factor
        self.mdx_scale = scale  # exposed so load-time warmup can align dummy frame counts to the time-downsampling stack
        f = config.audio.dim_f // self.num_subbands
        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)
        def encoder_block():
            nonlocal c, f
            out = _block(tfc_tdf=TFC_TDF(c, c, l, f, bn, norm, act), downscale=Downscale(c, c + g, scale, norm, act))
            f = f // scale[1]; c += g
            return out
        self.encoder_blocks = nn.ModuleList([encoder_block() for _ in range(n)])
        self.bottleneck_block = TFC_TDF(c, c, l, f, bn, norm, act)
        def decoder_block():
            nonlocal c, f
            upscale = Upscale(c, c - g, scale, norm, act)
            f = f * scale[1]; c -= g
            return _block(upscale=upscale, tfc_tdf=TFC_TDF(2 * c, c, l, f, bn, norm, act))
        self.decoder_blocks = nn.ModuleList([decoder_block() for _ in range(n)])
        self.final_conv = nn.Sequential(nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False), act, nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False))
        self.stft = SubbandSTFT(config.audio)
    def mlx_forward_mx(self, raw_audio):
        from .mdx23c_mlx import mlx_forward_mdx23c_mx
        return mlx_forward_mdx23c_mx(self, raw_audio, self.mps_model_compute_dtype)
    def _forward_core(self, x):
        encoder_outputs = []
        for block in self.encoder_blocks:
            encoder_outputs.append(x := block.tfc_tdf(x))
            x = block.downscale(x)
        x = self.bottleneck_block(x)
        for block in self.decoder_blocks:
            x = block.tfc_tdf(torch.cat([block.upscale(x), encoder_outputs.pop()], 1))
        return x
    def forward(self, x):
        if self._use_mlx_full_forward(x):
            try:
                from .mdx23c_mlx import mlx_forward_mdx23c
                return mlx_forward_mdx23c(self, x, self.mps_model_compute_dtype)
            except Exception as exc:
                self._pymss_mlx_full_backend_error = repr(exc)
                self.mps_model_backend = "torch"
        return forward_subband_mask_model(self, x, self._forward_core)