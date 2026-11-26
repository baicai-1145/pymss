"""NPU-level graph capture for the time-axis transformers (opt-in: S2_GRAPHS=time).

`torch.compile` on this model hits Dynamo graph breaks (custom autograd.Function for the dwconv, dict caches, the
62-band loops), and `make_graphed_callables` on the *whole* core fails on the 62-band frequency mixer (aclnnMatmul
161002: inner dim 62 is not 32B-aligned, which ACLGraph rejects although eager accepts it).  The six time-axis
transformers do capture, forward bit-identically (rel 0.0), and they hold the depthwise convs and the heavy matmuls.

Caveat measured with this module: a captured graph replays the RNG state of the capture, so dropout masks repeat.
`install(..., allow_frozen_dropout=True)` must therefore be an explicit decision, not a default.
"""
import os

import torch
from torch import nn


def enabled():
    return os.environ.get('S2_GRAPHS', '').lower() in ('time', '1', 'true', 'yes')


class GraphedTime(nn.Module):
    """Keeps the captured module as a submodule so its parameters stay visible to the optimizer/EMA."""

    def __init__(self, inner, graph):
        super().__init__()
        self.inner, self._graph = inner, graph

    def forward(self, x):
        return self._graph(x)


def install(model, sample, layers=None, allow_frozen_dropout=False, num_warmup_iters=3):
    """Capture each time-axis transformer at its static training shape and swap it in.

    Args:
        model: the arm model (``model.layers[i][0]`` are the time transformers).
        sample: one input tensor of the shape they receive, ``(b*f, t, d)``.
        layers: indices to capture (default: all).
        allow_frozen_dropout: required when the model is in training mode and uses dropout.
        num_warmup_iters: warmup iterations inside ``make_graphed_callables``.
    """
    idxs = range(len(model.layers)) if layers is None else layers
    amp = torch.npu.amp.autocast(dtype=torch.bfloat16, cache_enabled=False)  # reuse the instance: it is already a ctx
    captured = []
    for i in idxs:
        inner = model.layers[i][0]
        was_training = inner.training
        inner.train()
        with amp:
            graph = torch.npu.make_graphed_callables(inner, (sample,), num_warmup_iters=num_warmup_iters)
        inner.train(was_training)
        model.layers[i][0] = GraphedTime(inner, graph)
        captured.append(i)
    model._graphed_time = captured
    return captured


def frozen_dropout_risk(model):
    """True when a captured graph would freeze nonzero dropout masks."""
    rates = []
    for module in model.modules():
        p = getattr(module, 'dropout', None)
        if isinstance(p, (int, float)):
            rates.append(float(p))
    return max(rates) if rates else 0.0
