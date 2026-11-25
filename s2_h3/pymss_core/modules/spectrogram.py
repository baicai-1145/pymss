import torch
from torch import nn
from pymss_core.npu_fft import istft_npu, stft_npu, torch_stft_blocked
class SubbandSTFT:
    def __init__(self, config):
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = config.dim_f
    def __call__(self, x):
        b = x.shape[:-2]; c, l = x.shape[-2:]
        fft_fallback = torch_stft_blocked(x.device)
        stft_fn = stft_npu if fft_fallback else torch.stft
        flat = x.reshape(-1, l)
        if fft_fallback: flat = flat.float()
        x = torch.view_as_real(stft_fn(flat, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(x.device), center=True, return_complex=True))
        x = x.permute(0, 3, 1, 2)
        return x.reshape(*b, c * 2, -1, x.shape[-1])[..., : self.dim_f, :]
    def inverse(self, x):
        b = x.shape[:-3]; c, f, t = x.shape[-3:]
        full = self.n_fft // 2 + 1
        if full > f: x = torch.nn.functional.pad(x, (0, 0, 0, full - f))
        x = torch.view_as_complex(x.reshape(-1, 2, full, t).permute(0, 2, 3, 1).contiguous())
        istft_fn = istft_npu if torch_stft_blocked(x.device) else torch.istft
        return istft_fn(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(x.device), center=True).reshape([*b, 2, -1])
def get_activation(act_type):
    if act_type == "gelu": return nn.GELU()
    if act_type == "relu": return nn.ReLU()
    if act_type[:3] == "elu": return nn.ELU(float(act_type.replace("elu", "")))
    raise Exception
def cac_to_cws(x, num_subbands): return x.reshape(x.shape[0], x.shape[1] * num_subbands, x.shape[2] // num_subbands, x.shape[3])
def cws_to_cac(x, num_subbands): return x.reshape(x.shape[0], x.shape[1] // num_subbands, x.shape[2] * num_subbands, x.shape[3])
def forward_subband_mask_model(module, x, core_fn):
    mix = x = cac_to_cws(module.stft(x), module.num_subbands)
    first_conv_out = x = module.first_conv(x)
    x = core_fn(x.transpose(-1, -2)).transpose(-1, -2) * first_conv_out
    x = cws_to_cac(module.final_conv(torch.cat([mix, x], 1)), module.num_subbands)
    if module.num_target_instruments > 1: x = x.reshape(x.shape[0], module.num_target_instruments, -1, *x.shape[-2:])
    return module.stft.inverse(x)
