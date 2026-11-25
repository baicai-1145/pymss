from torch import nn
from torch.nn import Module, ModuleList
from .transformer import (Attention, FeedForward, RMSNorm, default_cuda_attention_backend, set_cuda_attention_backend, set_mps_attention_backend)
class _TransposeLast(Module):
    def forward(self, x): return x.transpose(1, 2)
class MacaronFF(Module):
    def __init__(self, dim, mult=4, dropout=0.0): super().__init__(); self.ff, self.scale = FeedForward(dim=dim, mult=mult, dropout=dropout), 0.5
    def forward(self, x): return self.ff(x) * self.scale
class ConformerConvModule(Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()
        inner = dim * expansion_factor
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        # Keep Sequential indices aligned with MSST checkpoints (Rearrange slots are parameter-free).
        self.net = nn.Sequential(RMSNorm(dim), _TransposeLast(), nn.Conv1d(dim, inner * 2, 1), nn.GLU(dim=1), nn.Conv1d(inner, inner, kernel_size, padding=(kernel_size - 1) // 2, groups=inner), nn.BatchNorm1d(inner), nn.SiLU(inplace=True), nn.Conv1d(inner, dim, 1), _TransposeLast(), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)
class ConformerBlock(Module):
    def __init__(self, *, dim, heads=8, dim_head=64, ff_mult=4, attn_dropout=0.0, ff_dropout=0.0, conv_expansion_factor=2, conv_kernel_size=31, rotary_embed=None, flash_attn=True, shared_qkv_bias=None, shared_out_bias=None):
        super().__init__()
        self.ff1, self.attn = MacaronFF(dim=dim, mult=ff_mult, dropout=ff_dropout), Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, shared_qkv_bias=shared_qkv_bias, shared_out_bias=shared_out_bias, rotary_embed=rotary_embed, flash=flash_attn)
        self.conv = ConformerConvModule(dim=dim, expansion_factor=conv_expansion_factor, kernel_size=conv_kernel_size, dropout=ff_dropout)
        self.ff2, self.out_norm = MacaronFF(dim=dim, mult=ff_mult, dropout=ff_dropout), RMSNorm(dim)
    def forward(self, x): x = x + self.ff1(x); x = x + self.attn(x); x = x + self.conv(x); x = x + self.ff2(x); return self.out_norm(x)
class Conformer(Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4, rotary_embed=None, flash_attn=True, conv_expansion_factor=2, conv_kernel_size=31, norm_output=True, shared_qkv_bias=None, shared_out_bias=None):
        super().__init__()
        self.layers = ModuleList([ConformerBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, attn_dropout=attn_dropout, ff_dropout=ff_dropout, conv_expansion_factor=conv_expansion_factor, conv_kernel_size=conv_kernel_size, rotary_embed=rotary_embed, flash_attn=flash_attn, shared_qkv_bias=shared_qkv_bias, shared_out_bias=shared_out_bias) for _ in range(depth)])
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()
        self.mps_attention_backend, self.mps_mlx_min_tokens, self.cuda_attention_backend = "torch", 128, default_cuda_attention_backend()
    def set_mps_attention_backend(self, backend=None, min_tokens=128): set_mps_attention_backend(self, backend, min_tokens, [block.attn for block in self.layers])
    def set_cuda_attention_backend(self, backend=None): set_cuda_attention_backend(self, backend, [block.attn for block in self.layers])
    def forward(self, x):
        for block in self.layers: x = block(x)
        return self.norm(x)