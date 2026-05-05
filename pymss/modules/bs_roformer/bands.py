from typing import Tuple

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from .transformer import RMSNorm


def default(v, d):
    return v if v is not None else d


def dim_input_offsets(dim_inputs):
    offsets = [0]
    for dim_input in dim_inputs:
        offsets.append(offsets[-1] + dim_input)
    return tuple(offsets)


def contiguous_dim_groups(dim_inputs):
    groups = []
    start = 0
    for i in range(1, len(dim_inputs) + 1):
        if i == len(dim_inputs) or dim_inputs[i] != dim_inputs[start]:
            groups.append((start, i, dim_inputs[start]))
            start = i
    return tuple(groups)


def grouped_linear(x, weight, bias):
    group_count, out_features, in_features = weight.shape
    leading_shape = x.shape[:-2]
    x = x.reshape(-1, group_count, in_features).transpose(0, 1)
    out = torch.bmm(x, weight.transpose(1, 2))
    out = out.transpose(0, 1).reshape(*leading_shape, group_count, out_features)
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out


class BandSplit(Module):
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self._dim_offsets = dim_input_offsets(dim_inputs)
        self._dim_groups = contiguous_dim_groups(dim_inputs)
        self._group_cache = {}
        self.use_grouped_forward = True
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            self.to_features.append(nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            ))

    def _get_group_params(self, start, end, device, dtype):
        key = (start, end, device.type, device.index, dtype)
        cached = self._group_cache.get(key)
        if cached is not None:
            return cached

        norms = [self.to_features[i][0] for i in range(start, end)]
        linears = [self.to_features[i][1] for i in range(start, end)]
        gamma = torch.stack([norm.gamma.to(device=device, dtype=dtype) for norm in norms], dim=0)
        weight = torch.stack([linear.weight.to(device=device, dtype=dtype) for linear in linears], dim=0)
        bias = None
        if linears[0].bias is not None:
            bias = torch.stack([linear.bias.to(device=device, dtype=dtype) for linear in linears], dim=0)

        cached = (gamma, weight, bias)
        self._group_cache[key] = cached
        return cached

    def _forward_grouped(self, x):
        outs = []
        for start, end, dim_in in self._dim_groups:
            offset_start = self._dim_offsets[start]
            offset_end = self._dim_offsets[end]
            group_x = x[..., offset_start:offset_end].reshape(*x.shape[:-1], end - start, dim_in)
            gamma, weight, bias = self._get_group_params(start, end, x.device, x.dtype)
            group_x = F.normalize(group_x, dim=-1) * (dim_in ** 0.5) * gamma
            outs.append(grouped_linear(group_x, weight, bias))
        return torch.cat(outs, dim=-2)

    def warm_group_cache(self, device, dtype):
        for start, end, _ in self._dim_groups:
            self._get_group_params(start, end, device, dtype)

    def forward(self, x):
        if not self.training and self.use_grouped_forward:
            return self._forward_grouped(x)

        outs = []
        for split_input, to_feature in zip(x.split(self.dim_inputs, dim=-1), self.to_features):
            outs.append(to_feature(split_input))
        return torch.stack(outs, dim=-2)


def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh,
        hidden_layers=None,
):
    dim_hidden = default(dim_hidden, dim_in)
    hidden_layers = default(hidden_layers, max(depth - 1, 0))
    dims = (dim_in, *((dim_hidden,) * hidden_layers), dim_out)
    net = []

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)
        net.append(nn.Linear(layer_dim_in, layer_dim_out))
        if not is_last:
            net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4,
            mlp_hidden_layers=None,
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self._dim_groups = contiguous_dim_groups(dim_inputs)
        self._group_cache = {}
        self.use_grouped_forward = True
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            self.to_freqs.append(nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth, hidden_layers=mlp_hidden_layers),
                nn.GLU(dim=-1)
            ))

    def _groupable_layers(self, mlp_with_glu):
        if not isinstance(mlp_with_glu, nn.Sequential) or len(mlp_with_glu) != 2:
            return None
        mlp, glu = mlp_with_glu
        if not isinstance(glu, nn.GLU) or not isinstance(mlp, nn.Sequential):
            return None

        layers = []
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                layers.append(('linear', layer))
            elif isinstance(layer, nn.Tanh):
                layers.append(('tanh', None))
            else:
                return None
        if not layers or layers[-1][0] != 'linear':
            return None
        return tuple(layers)

    def _can_group_mlp(self):
        base_signature = None
        for mlp_with_glu in self.to_freqs:
            layers = self._groupable_layers(mlp_with_glu)
            if layers is None:
                return False
            signature = tuple(
                item if kind != 'linear' else (kind, item.in_features, item.out_features, item.bias is not None)
                for kind, item in layers
            )
            if base_signature is None:
                base_signature = signature
            elif signature != base_signature:
                return False
        return True

    def _get_group_params(self, start, end, device, dtype):
        key = (start, end, device.type, device.index, dtype)
        cached = self._group_cache.get(key)
        if cached is not None:
            return cached

        grouped_layers = []
        first_layers = self._groupable_layers(self.to_freqs[start])
        for layer_index, (kind, _) in enumerate(first_layers):
            if kind == 'tanh':
                grouped_layers.append(('tanh', None, None))
                continue

            linears = [self._groupable_layers(self.to_freqs[i])[layer_index][1] for i in range(start, end)]
            weight = torch.stack([linear.weight.to(device=device, dtype=dtype) for linear in linears], dim=0)
            bias = None
            if linears[0].bias is not None:
                bias = torch.stack([linear.bias.to(device=device, dtype=dtype) for linear in linears], dim=0)
            grouped_layers.append(('linear', weight, bias))

        cached = tuple(grouped_layers)
        self._group_cache[key] = cached
        return cached

    def _forward_grouped_mlp(self, x):
        outs = []
        for start, end, _ in self._dim_groups:
            group_x = x[:, :, start:end, :]
            for kind, weight, bias in self._get_group_params(start, end, x.device, x.dtype):
                if kind == 'linear':
                    group_x = grouped_linear(group_x, weight, bias)
                else:
                    group_x = torch.tanh(group_x)
            outs.append(F.glu(group_x, dim=-1).flatten(start_dim=-2))
        return torch.cat(outs, dim=-1)

    def forward(self, x):
        if not self.training and self.use_grouped_forward and self._can_group_mlp():
            return self._forward_grouped_mlp(x)

        outs = []
        for band_features, mlp in zip(x.unbind(dim=-2), self.to_freqs):
            outs.append(mlp(band_features))
        return torch.cat(outs, dim=-1)

    def warm_group_cache(self, device, dtype):
        if not self._can_group_mlp():
            return
        for start, end, _ in self._dim_groups:
            self._get_group_params(start, end, device, dtype)
