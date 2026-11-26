"""Ascend multi-tensor optimizer + fused EMA for the training loop (opt-in, `S2_OPT=fused`).

`torch_npu.optim.NpuFusedAdamW` folds ~500 per-parameter optimizer launches into one or two CANN kernels. Two things
block it in this codebase and both are handled here:

* it is **not** a subclass of `torch.optim.AdamW` (MRO: NpuFusedAdamW -> NpuFusedOptimizerBase -> Optimizer), so the
  startup assertion in `s2_aw_entry.py` must accept it explicitly;
* `zero_grad(set_to_none=True)` raises `ValueError: set_to_none is not supported in fused optimizers`, and the training
  loop calls it that way — the subclass below ignores the flag (a fused optimizer keeps its grads as a flat buffer).

`get_combined_params()` exposes one flat fp32 buffer, so the EMA can be a single `lerp_` instead of ~550 launches.
"""
import torch


def make_optimizer(params, lr, weight_decay, kind='default'):
    if kind != 'fused':
        return None
    import torch_npu
    return FixedNpuFusedAdamW(params, lr=lr, weight_decay=weight_decay)


def _base_class():
    import torch_npu
    fused = getattr(getattr(torch_npu, 'optim', None), 'NpuFusedAdamW', None)
    if fused is None:
        return None
    return fused


class _FusedMixin:
    def zero_grad(self, set_to_none: bool = True):
        # The fused optimizer owns one flat gradient buffer; the flag has no meaning and raises if forwarded.
        return super().zero_grad()


_base = _base_class()
FixedNpuFusedAdamW = type('FixedNpuFusedAdamW', (_FusedMixin, _base), {}) if _base is not None else None


class FlatEMA:
    """EMA over the optimizer's combined fp32 parameter buffer: one `lerp_` per step instead of ~550."""

    @staticmethod
    def _flat(optimizer):
        combined = optimizer.get_combined_params()
        if isinstance(combined, (list, tuple)):
            combined = next((c for c in combined if c is not None), None)
        return combined

    def __init__(self, optimizer, decay=0.999):
        self.decay, self.n_averaged = 1.0 - decay, 0
        flat = self._flat(optimizer)
        # the fused optimizer only materialises its combined buffer after the first step
        self.shadow = None if flat is None else flat.detach().clone()

    def update_parameters(self, model=None):
        optimizer = getattr(self, '_opt', None)
        if optimizer is None:
            raise RuntimeError('bind the optimizer first: FlatEMA.attach(opt)')
        flat = self._flat(optimizer)
        if self.shadow is None:
            self.shadow = flat.detach().clone()
        else:
            self.shadow.lerp_(flat.detach(), 1.0 - self.decay)
        self.n_averaged += 1

    def attach(self, optimizer):
        self._opt = optimizer
        return self
