import torch.nn.functional as F
from torch import einsum, nn
class Attend(nn.Module):
    def __init__(self, dropout=0.0, flash=False, scale=None): super().__init__(); self.scale,self.dropout,self.attn_dropout,self.flash = scale, dropout, nn.Dropout(dropout), flash
    def forward(self, q, k, v):
        if self.flash: return F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=self.dropout if self.training else 0.0)
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * (self.scale if self.scale is not None else q.shape[-1] ** -0.5)
        return einsum("b h i j, b h j d -> b h i d", self.attn_dropout(sim.softmax(dim=-1)), v)