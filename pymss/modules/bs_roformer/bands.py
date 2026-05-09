from collections import defaultdict
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
    weight = weight.transpose(1, 2)
    if bias is None:
        out = torch.bmm(x, weight)
    else:
        bias = bias.to(dtype=x.dtype).unsqueeze(1).expand(-1, x.shape[1], -1)
        out = torch.baddbmm(bias, x, weight)
    out = out.transpose(0, 1).reshape(*leading_shape, group_count, out_features)
    return out


def inference_tanh(x):
    if torch.is_grad_enabled():
        return torch.tanh(x)
    return x.tanh_()


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
        self._dim_offsets = dim_input_offsets(dim_inputs)
        self._dim_groups = contiguous_dim_groups(dim_inputs)
        self._group_cache = {}
        self._layer_group_cache = {}
        self._index_cache = {}
        self._packed_layer_group_cache = {}
        self._layer_group_plan = None
        self._layer_group_plan_ready = False
        self._can_group_mlp_cache = None
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

    def _layer_grouping_plan(self):
        if self._layer_group_plan_ready:
            return self._layer_group_plan

        band_layers = [self._groupable_layers(mlp_with_glu) for mlp_with_glu in self.to_freqs]
        if any(layers is None for layers in band_layers):
            self._layer_group_plan_ready = True
            self._layer_group_plan = None
            return None

        layer_count = len(band_layers[0])
        if any(len(layers) != layer_count for layers in band_layers):
            self._layer_group_plan_ready = True
            self._layer_group_plan = None
            return None

        plan = []
        for layer_index in range(layer_count):
            first_kind = band_layers[0][layer_index][0]
            if first_kind == 'tanh':
                if any(layers[layer_index][0] != 'tanh' for layers in band_layers):
                    self._layer_group_plan_ready = True
                    self._layer_group_plan = None
                    return None
                plan.append(('tanh', None))
                continue

            if first_kind != 'linear':
                self._layer_group_plan_ready = True
                self._layer_group_plan = None
                return None

            groups = defaultdict(list)
            for band_index, layers in enumerate(band_layers):
                kind, layer = layers[layer_index]
                if kind != 'linear':
                    self._layer_group_plan_ready = True
                    self._layer_group_plan = None
                    return None
                signature = (layer.in_features, layer.out_features, layer.bias is not None)
                groups[signature].append(band_index)

            plan.append(('linear', tuple((signature, tuple(indices)) for signature, indices in groups.items())))

        # Extra hidden layers in MBR produce large per-band hidden->hidden
        # batched GEMMs. On CUDA those are slower than the existing addmm
        # loop, so keep this fast path to the common two-linear mask heads.
        if sum(1 for kind, _ in plan if kind == 'linear') > 2:
            self._layer_group_plan_ready = True
            self._layer_group_plan = None
            return None

        self._layer_group_plan_ready = True
        self._layer_group_plan = tuple(plan)
        return self._layer_group_plan

    def _can_group_mlp(self):
        if self._can_group_mlp_cache is not None:
            return self._can_group_mlp_cache

        base_signature = None
        for mlp_with_glu in self.to_freqs:
            layers = self._groupable_layers(mlp_with_glu)
            if layers is None:
                self._can_group_mlp_cache = False
                return False
            signature = tuple(
                item if kind != 'linear' else (kind, item.in_features, item.out_features, item.bias is not None)
                for kind, item in layers
            )
            if base_signature is None:
                base_signature = signature
            elif signature != base_signature:
                self._can_group_mlp_cache = False
                return False
        self._can_group_mlp_cache = True
        return True

    def _indices_tensor(self, indices, device):
        key = (indices, device.type, device.index)
        cached = self._index_cache.get(key)
        if cached is None or cached.device != device:
            cached = torch.as_tensor(indices, device=device)
            self._index_cache[key] = cached
        return cached

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

    def _get_layer_group_params(self, layer_index, signature, indices, device, dtype):
        key = (layer_index, signature, indices, device.type, device.index, dtype)
        cached = self._layer_group_cache.get(key)
        if cached is not None:
            return cached

        linears = [self._groupable_layers(self.to_freqs[i])[layer_index][1] for i in indices]
        weight = torch.stack([linear.weight.to(device=device, dtype=dtype) for linear in linears], dim=0)
        bias = None
        if linears[0].bias is not None:
            bias = torch.stack([linear.bias.to(device=device, dtype=dtype) for linear in linears], dim=0)

        cached = (weight, bias)
        self._layer_group_cache[key] = cached
        return cached

    def _get_packed_layer_group_params(self, estimators, layer_index, signature, indices, device, dtype):
        estimator_ids = tuple(id(estimator) for estimator in estimators)
        key = (estimator_ids, layer_index, signature, indices, device.type, device.index, dtype)
        cached = self._packed_layer_group_cache.get(key)
        if cached is not None:
            return cached

        linears = [
            estimator._groupable_layers(estimator.to_freqs[band_index])[layer_index][1]
            for estimator in estimators
            for band_index in indices
        ]
        weight = torch.stack([linear.weight.to(device=device, dtype=dtype) for linear in linears], dim=0)
        bias = None
        if linears[0].bias is not None:
            bias = torch.stack([linear.bias.to(device=device, dtype=dtype) for linear in linears], dim=0)

        cached = (weight, bias)
        self._packed_layer_group_cache[key] = cached
        return cached

    @staticmethod
    def _packable_estimators_by_band(estimators):
        estimators = tuple(estimators)
        if len(estimators) <= 1:
            return False

        first = estimators[0]
        if not isinstance(first, MaskEstimator):
            return False
        if first.training or not first.use_grouped_forward:
            return False

        first_layers = [first._groupable_layers(mlp_with_glu) for mlp_with_glu in first.to_freqs]
        if any(layers is None for layers in first_layers):
            return False
        first_signatures = tuple(
            tuple(item if kind != 'linear' else (kind, item.in_features, item.out_features, item.bias is not None)
                  for kind, item in layers)
            for layers in first_layers
        )

        for estimator in estimators[1:]:
            if type(estimator) is not type(first):
                return False
            if estimator.training or not estimator.use_grouped_forward:
                return False
            if estimator.dim_inputs != first.dim_inputs:
                return False

            layers = [estimator._groupable_layers(mlp_with_glu) for mlp_with_glu in estimator.to_freqs]
            if any(layer_group is None for layer_group in layers):
                return False
            signatures = tuple(
                tuple(item if kind != 'linear' else (kind, item.in_features, item.out_features, item.bias is not None)
                      for kind, item in layer_group)
                for layer_group in layers
            )
            if signatures != first_signatures:
                return False
        return True

    def _forward_grouped_mlp(self, x):
        outs = []
        for start, end, _ in self._dim_groups:
            group_x = x[:, :, start:end, :]
            for kind, weight, bias in self._get_group_params(start, end, x.device, x.dtype):
                if kind == 'linear':
                    group_x = grouped_linear(group_x, weight, bias)
                else:
                    group_x = inference_tanh(group_x)
            outs.append(F.glu(group_x, dim=-1).flatten(start_dim=-2))
        return torch.cat(outs, dim=-1)

    def _forward_layer_grouped_mlp(self, x):
        plan = self._layer_grouping_plan()
        if plan is None:
            return None

        group_x = x
        for layer_index, (kind, groups) in enumerate(plan):
            if kind == 'tanh':
                group_x = inference_tanh(group_x)
                continue

            out_dims = {signature[1] for signature, _ in groups}
            if len(out_dims) != 1:
                if layer_index != len(plan) - 1:
                    return None
                outs = [None] * len(self.to_freqs)
                for signature, indices in groups:
                    weight, bias = self._get_layer_group_params(layer_index, signature, indices, x.device, x.dtype)
                    band_index = self._indices_tensor(indices, x.device)
                    selected = group_x.index_select(-2, band_index)
                    out = F.glu(grouped_linear(selected, weight, bias), dim=-1)
                    for band_position, band_out in zip(indices, out.unbind(dim=-2)):
                        outs[band_position] = band_out
                return torch.cat(outs, dim=-1)

            next_x = group_x.new_empty(*group_x.shape[:-1], next(iter(out_dims)))
            for signature, indices in groups:
                weight, bias = self._get_layer_group_params(layer_index, signature, indices, x.device, x.dtype)
                band_index = self._indices_tensor(indices, x.device)
                selected = group_x.index_select(-2, band_index)
                out = grouped_linear(selected, weight, bias)
                next_x.index_copy_(-2, band_index, out)
            group_x = next_x

        return F.glu(group_x, dim=-1).flatten(start_dim=-2)

    def _forward_by_band_fast(self, x):
        outs = []
        for band_index, band_features in enumerate(x.unbind(dim=-2)):
            group_x = band_features
            layers = self._groupable_layers(self.to_freqs[band_index])
            if layers is None:
                return None

            for kind, layer in layers:
                if kind == 'tanh':
                    group_x = inference_tanh(group_x)
                else:
                    group_x = layer(group_x)
            outs.append(F.glu(group_x, dim=-1))

        return torch.cat(outs, dim=-1)

    @staticmethod
    def _packable_estimators(estimators):
        estimators = tuple(estimators)
        if len(estimators) <= 1:
            return False

        first = estimators[0]
        if not isinstance(first, MaskEstimator):
            return False
        if first.training or not first.use_grouped_forward:
            return False

        first_plan = first._layer_grouping_plan()
        if first_plan is None:
            return False

        for estimator in estimators[1:]:
            if type(estimator) is not type(first):
                return False
            if estimator.training or not estimator.use_grouped_forward:
                return False
            if estimator.dim_inputs != first.dim_inputs:
                return False
            if estimator._layer_grouping_plan() != first_plan:
                return False
        return True

    @staticmethod
    def _select_packed_group(group_x, band_index, stem_count):
        if group_x.ndim == 4:
            selected = group_x.index_select(-2, band_index)
            return selected.unsqueeze(2).expand(-1, -1, stem_count, -1, -1)
        return group_x.index_select(-2, band_index)

    @staticmethod
    def _forward_packed_estimators_two_layer_stream(estimators, x, plan):
        if len(plan) != 3 or plan[0][0] != 'linear' or plan[1][0] != 'tanh' or plan[2][0] != 'linear':
            return None

        first = estimators[0]
        first_groups = plan[0][1]
        final_groups = plan[2][1]
        if len(first_groups) != 1:
            return None

        first_signature, _ = first_groups[0]
        stem_count = len(estimators)
        result = x.new_empty(x.shape[0], stem_count, x.shape[1], sum(first.dim_inputs))

        for final_signature, indices in final_groups:
            weight, bias = first._get_packed_layer_group_params(
                estimators, 0, first_signature, indices, x.device, x.dtype
            )
            band_index = first._indices_tensor(indices, x.device)
            selected = x.index_select(-2, band_index).unsqueeze(2).expand(-1, -1, stem_count, -1, -1)
            b, t, s, g, d = selected.shape
            group_x = grouped_linear(selected.reshape(b, t, s * g, d), weight, bias)
            group_x = inference_tanh(group_x)

            weight, bias = first._get_packed_layer_group_params(
                estimators, 2, final_signature, indices, x.device, x.dtype
            )
            group_x = grouped_linear(group_x, weight, bias)
            group_x = F.glu(group_x, dim=-1).reshape(b, t, s, g, -1)

            if indices == tuple(range(indices[0], indices[-1] + 1)):
                offset_start = first._dim_offsets[indices[0]]
                offset_end = first._dim_offsets[indices[-1] + 1]
                result[:, :, :, offset_start:offset_end] = group_x.flatten(start_dim=-2).permute(0, 2, 1, 3)
            else:
                for group_position, band_position in enumerate(indices):
                    offset_start = first._dim_offsets[band_position]
                    offset_end = first._dim_offsets[band_position + 1]
                    result[:, :, :, offset_start:offset_end] = group_x[:, :, :, group_position, :].permute(0, 2, 1, 3)

        return result

    @staticmethod
    def forward_packed_estimators(estimators, x):
        estimators = tuple(estimators)
        if not MaskEstimator._packable_estimators(estimators):
            return MaskEstimator._forward_packed_estimators_by_band(estimators, x)

        first = estimators[0]
        plan = first._layer_grouping_plan()
        streamed = MaskEstimator._forward_packed_estimators_two_layer_stream(estimators, x, plan)
        if streamed is not None:
            return streamed

        stem_count = len(estimators)
        band_count = len(first.to_freqs)
        group_x = x

        for layer_index, (kind, groups) in enumerate(plan):
            if kind == 'tanh':
                group_x = inference_tanh(group_x)
                continue

            out_dims = {signature[1] for signature, _ in groups}
            if len(out_dims) != 1:
                if layer_index != len(plan) - 1:
                    return None

                result = x.new_empty(x.shape[0], x.shape[1], stem_count, sum(first.dim_inputs))
                for signature, indices in groups:
                    weight, bias = first._get_packed_layer_group_params(
                        estimators, layer_index, signature, indices, x.device, x.dtype
                    )
                    band_index = first._indices_tensor(indices, x.device)
                    selected = MaskEstimator._select_packed_group(group_x, band_index, stem_count)
                    b, t, s, g, d = selected.shape
                    out = grouped_linear(selected.reshape(b, t, s * g, d), weight, bias)
                    out = F.glu(out, dim=-1).reshape(b, t, s, g, -1)
                    if indices == tuple(range(indices[0], indices[-1] + 1)):
                        offset_start = first._dim_offsets[indices[0]]
                        offset_end = first._dim_offsets[indices[-1] + 1]
                        result[:, :, :, offset_start:offset_end] = out.flatten(start_dim=-2)
                    else:
                        for group_position, band_position in enumerate(indices):
                            offset_start = first._dim_offsets[band_position]
                            offset_end = first._dim_offsets[band_position + 1]
                            result[:, :, :, offset_start:offset_end] = out[:, :, :, group_position, :]

                return result.permute(0, 2, 1, 3)

            out_dim = next(iter(out_dims))
            next_x = x.new_empty(x.shape[0], x.shape[1], stem_count, band_count, out_dim)
            for signature, indices in groups:
                weight, bias = first._get_packed_layer_group_params(
                    estimators, layer_index, signature, indices, x.device, x.dtype
                )
                band_index = first._indices_tensor(indices, x.device)
                selected = MaskEstimator._select_packed_group(group_x, band_index, stem_count)
                b, t, s, g, d = selected.shape
                out = grouped_linear(selected.reshape(b, t, s * g, d), weight, bias)
                next_x.index_copy_(-2, band_index, out.reshape(b, t, s, g, out_dim))
            group_x = next_x

        out = F.glu(group_x, dim=-1).flatten(start_dim=-2)
        return out.permute(0, 2, 1, 3)

    @staticmethod
    def _forward_packed_estimators_by_band(estimators, x):
        estimators = tuple(estimators)
        if not MaskEstimator._packable_estimators_by_band(estimators):
            return None

        first = estimators[0]
        stem_count = len(estimators)
        result = x.new_empty(x.shape[0], stem_count, x.shape[1], sum(first.dim_inputs))

        for band_index, mlp_with_glu in enumerate(first.to_freqs):
            group_x = x[:, :, band_index, :].unsqueeze(-2).expand(-1, -1, stem_count, -1)
            for layer_index, (kind, layer) in enumerate(first._groupable_layers(mlp_with_glu)):
                if kind == 'tanh':
                    group_x = inference_tanh(group_x)
                    continue

                signature = (layer.in_features, layer.out_features, layer.bias is not None)
                weight, bias = first._get_packed_layer_group_params(
                    estimators, layer_index, signature, (band_index,), x.device, x.dtype
                )
                group_x = grouped_linear(group_x, weight, bias)

            offset_start = first._dim_offsets[band_index]
            offset_end = first._dim_offsets[band_index + 1]
            result[:, :, :, offset_start:offset_end] = F.glu(group_x, dim=-1).permute(0, 2, 1, 3)

        return result

    def forward(self, x):
        if not self.training and self.use_grouped_forward:
            if self._can_group_mlp():
                return self._forward_grouped_mlp(x)
            grouped = self._forward_layer_grouped_mlp(x)
            if grouped is not None:
                return grouped
            by_band = self._forward_by_band_fast(x)
            if by_band is not None:
                return by_band

        outs = []
        for band_features, mlp in zip(x.unbind(dim=-2), self.to_freqs):
            outs.append(mlp(band_features))
        return torch.cat(outs, dim=-1)

    def warm_group_cache(self, device, dtype):
        if self._can_group_mlp():
            for start, end, _ in self._dim_groups:
                self._get_group_params(start, end, device, dtype)
            return

        plan = self._layer_grouping_plan()
        if plan is None:
            return
        for layer_index, (kind, groups) in enumerate(plan):
            if kind == 'tanh':
                continue
            for signature, indices in groups:
                self._get_layer_group_params(layer_index, signature, indices, device, dtype)
