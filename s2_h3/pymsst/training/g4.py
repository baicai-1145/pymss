"""Opt-in training kernels. Leave parameter names, dropout and eval intact."""
import torch
import torch.nn.functional as F
from pymss_core.modules.bs_roformer.transformer import FeedForward, RMSNorm


@torch.compile(backend="inductor", fullgraph=True)
def _normalized(x, scale, gamma):
    # Match the core clamp on the L2 norm (not rsqrt(mean(x*x) + eps)).
    return F.normalize(x, dim=-1) * scale * gamma


class G4RMSNorm(RMSNorm):
    def forward(self, x):
        if not self.training or not x.is_cuda or not torch.is_grad_enabled():
            return super().forward(x)
        return _normalized(x, self.scale, self.gamma)


@torch.compile(backend="inductor", fullgraph=True)
def _normalized_bf16(x, scale, gamma):
    return (F.normalize(x, dim=-1) * scale * gamma).to(torch.bfloat16)


class G4NormForLinear(G4RMSNorm):
    def forward(self, x):
        if self.training and torch.is_grad_enabled() and x.is_cuda and torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') == torch.bfloat16:
            return _normalized_bf16(x, self.scale, self.gamma)
        return super().forward(x)


@torch.compile(backend="inductor", fullgraph=True)
def _gelu_dropout_backward(grad, x, mask, p):
    grad = torch.ops.aten.native_dropout_backward.default(grad, mask, 1.0 / (1.0 - p))
    return torch.ops.aten.gelu_backward.default(grad, x, approximate="none")


class _GeluDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        out, mask = torch.ops.aten.native_dropout.default(F.gelu(x), p, True)
        ctx.save_for_backward(x, mask)
        ctx.p = p
        return out

    @staticmethod
    def backward(ctx, grad):
        x, mask = ctx.saved_tensors
        return _gelu_dropout_backward(grad, x, mask, ctx.p), None


class G4FeedForward(FeedForward):
    def forward(self, x):
        norm, up, _, drop, down, last = self.net
        if not (self.training and drop.training and 0 < drop.p < 1 and torch.is_grad_enabled()
                and x.is_cuda and torch.is_autocast_enabled('cuda')
                and torch.get_autocast_dtype('cuda') == torch.bfloat16):
            return super().forward(x)
        return last(down(_GeluDropout.apply(up(norm(x)), drop.p)))


def install_g4(model):
    width = getattr(getattr(model, 'final_norm', None), 'gamma', None)
    norms = 0
    for mod in model.modules():
        if type(mod) is RMSNorm and width is not None and mod.gamma.shape == width.shape:
            mod.__class__ = G4RMSNorm
            norms += 1
    # Only narrow where *all* consumers immediately apply autocast linear layers.
    # Leave the residual stream and final normalization in their original dtype.
    from .g4_rope import PackedAttention
    targets = []
    for mod in model.modules():
        if type(mod) is FeedForward and type(mod.net[1]) is torch.nn.Linear:
            targets.append(mod.net[0])
            if (len(mod.net) == 6 and type(mod.net[2]) is torch.nn.GELU
                    and mod.net[2].approximate == 'none' and type(mod.net[3]) is torch.nn.Dropout
                    and not mod.net[3].inplace and 0 < mod.net[3].p < 1):
                mod.__class__ = G4FeedForward
        elif type(mod) is PackedAttention:
            targets.append(mod.norm)
    temporal = getattr(model, 'temporal_norms', ())
    convs = getattr(model, 'convs', ())
    if getattr(model, 'arm', None) == 'F' and len(temporal) == len(convs):
        targets.extend(temporal)
    for norm in targets:
        if type(norm) is G4RMSNorm:
            norm.__class__ = G4NormForLinear
    return norms
