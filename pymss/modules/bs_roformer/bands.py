from collections import defaultdict
from itertools import accumulate
from typing import Tuple

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from .triton_kernels import mask_final_tanh_glu_flat_triton, mask_final_tanh_glu_packed_triton, triton_kernels_available
from .transformer import RMSNorm


def default(v, d):
    return v if v is not None else d


def dim_input_offsets(dim_inputs):
    return (0, *accumulate(dim_inputs))


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
        if bias.dtype != x.dtype or bias.device != x.device:
            bias = bias.to(device=x.device, dtype=x.dtype)
        bias = bias.unsqueeze(1).expand(-1, x.shape[1], -1)
        out = torch.baddbmm(bias, x, weight)
    return out.transpose(0, 1).reshape(*leading_shape, group_count, out_features)


def inference_tanh(x):
    return torch.tanh(x) if torch.is_grad_enabled() else x.tanh_()


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
        self.to_features = ModuleList([nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim)) for dim_in in dim_inputs])

    def _get_group_params(self, start, end, device, dtype):
        key = (start, end, device.type, device.index, dtype)
        cached = self._group_cache.get(key)
        if cached is not None:
            return cached

        norms = [self.to_features[i][0] for i in range(start, end)]
        linears = [self.to_features[i][1] for i in range(start, end)]
        gamma = torch.stack([norm.gamma.to(device=device, dtype=dtype) for norm in norms], dim=0)
        weight = torch.stack([linear.weight.to(device=device, dtype=dtype) for linear in linears], dim=0)
        cached = (
            gamma,
            weight,
            torch.stack([linear.bias.to(device=device, dtype=dtype) for linear in linears], dim=0) if linears[0].bias is not None else None,
        )
        self._group_cache[key] = cached
        return cached

    def _forward_grouped(self, x):
        return torch.cat([self._forward_group(x, start, end, dim_in) for start, end, dim_in in self._dim_groups], dim=-2)

    def _forward_group(self, x, start, end, dim_in):
        offset_start = self._dim_offsets[start]
        offset_end = self._dim_offsets[end]
        group_x = x[..., offset_start:offset_end].reshape(*x.shape[:-1], end - start, dim_in)
        gamma, weight, bias = self._get_group_params(start, end, x.device, x.dtype)
        group_x = F.normalize(group_x, dim=-1) * (dim_in ** 0.5) * gamma
        return grouped_linear(group_x, weight, bias)

    def warm_group_cache(self, device, dtype):
        for start, end, _ in self._dim_groups:
            self._get_group_params(start, end, device, dtype)

    def forward(self, x):
        if not self.training and self.use_grouped_forward:
            return self._forward_grouped(x)

        return torch.stack([to_feature(split_input) for split_input, to_feature in zip(x.split(self.dim_inputs, dim=-1), self.to_features)], dim=-2)


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
    return nn.Sequential(*[
        layer
        for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:]))
        for layer in ((nn.Linear(layer_dim_in, layer_dim_out),) if ind == len(dims) - 2 else (nn.Linear(layer_dim_in, layer_dim_out), activation()))
    ])


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
        self._dim_total = sum(dim_inputs)
        self._dim_offsets = dim_input_offsets(dim_inputs)
        self._dim_groups = contiguous_dim_groups(dim_inputs)
        self._group_cache = {}
        self._layer_group_cache = {}
        self._index_cache = {}
        self._packed_layer_group_cache = {}
        self._groupable_layers_cache = {}
        self._band_layers_cache = None
        self._band_signatures_cache = None
        self._layer_group_plan = None
        self._layer_group_plan_ready = False
        self._can_group_mlp_cache = None
        self.use_grouped_forward = True
        dim_hidden = dim * mlp_expansion_factor
        self.to_freqs = ModuleList([
            nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth, hidden_layers=mlp_hidden_layers),
                nn.GLU(dim=-1)
            )
            for dim_in in dim_inputs
        ])

    def _groupable_layers(self, mlp_with_glu):
        cache_key = id(mlp_with_glu)
        if cache_key in self._groupable_layers_cache:
            return self._groupable_layers_cache[cache_key]

        if not isinstance(mlp_with_glu, nn.Sequential) or len(mlp_with_glu) != 2:
            self._groupable_layers_cache[cache_key] = None
            return None
        mlp, glu = mlp_with_glu
        if not isinstance(glu, nn.GLU) or not isinstance(mlp, nn.Sequential):
            self._groupable_layers_cache[cache_key] = None
            return None

        layers = []
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                layers.append(('linear', layer))
            elif isinstance(layer, nn.Tanh):
                layers.append(('tanh', None))
            else:
                self._groupable_layers_cache[cache_key] = None
                return None
        if not layers or layers[-1][0] != 'linear':
            self._groupable_layers_cache[cache_key] = None
            return None
        self._groupable_layers_cache[cache_key] = layers = tuple(layers)
        return layers

    def _band_groupable_layers(self):
        if self._band_layers_cache is None:
            self._band_layers_cache = tuple(self._groupable_layers(mlp_with_glu) for mlp_with_glu in self.to_freqs)
        return self._band_layers_cache

    @staticmethod
    def _layers_signature(layers):
        return tuple(item if kind != 'linear' else (kind, item.in_features, item.out_features, item.bias is not None) for kind, item in layers)

    def _band_layer_signatures(self):
        if self._band_signatures_cache is None:
            layers = self._band_groupable_layers()
            self._band_signatures_cache = None if any(layer_group is None for layer_group in layers) else tuple(self._layers_signature(layer_group) for layer_group in layers)
        return self._band_signatures_cache

    def _layer_grouping_plan(self):
        if self._layer_group_plan_ready:
            return self._layer_group_plan

        band_layers = self._band_groupable_layers()
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

        signatures = self._band_layer_signatures()
        self._can_group_mlp_cache = signatures is not None and all(signature == signatures[0] for signature in signatures)
        return True

    def _indices_tensor(self, indices, device):
        key = (indices, device.type, device.index)
        cached = self._index_cache.get(key)
        if cached is None or cached.device != device:
            cached = torch.as_tensor(indices, device=device)
            self._index_cache[key] = cached
        return cached

    @staticmethod
    def _contiguous_bounds(indices):
        return (indices[0], indices[-1] + 1) if indices and indices == tuple(range(indices[0], indices[-1] + 1)) else None

    def _select_group(self, group_x, indices):
        bounds = self._contiguous_bounds(indices)
        return group_x[..., bounds[0]:bounds[1], :] if bounds is not None else group_x.index_select(-2, self._indices_tensor(indices, group_x.device))

    def _copy_group_(self, dst, indices, src):
        bounds = self._contiguous_bounds(indices)
        if bounds is None:
            dst.index_copy_(-2, self._indices_tensor(indices, dst.device), src)
        else:
            dst[..., bounds[0]:bounds[1], :] = src

    def _get_group_params(self, start, end, device, dtype):
        key = (start, end, device.type, device.index, dtype)
        cached = self._group_cache.get(key)
        if cached is not None:
            return cached

        grouped_layers = []
        band_layers = self._band_groupable_layers()
        first_layers = band_layers[start]
        for layer_index, (kind, _) in enumerate(first_layers):
            if kind == 'tanh':
                grouped_layers.append(('tanh', None, None))
                continue

            linears = [band_layers[i][layer_index][1] for i in range(start, end)]
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

        band_layers = self._band_groupable_layers()
        linears = [band_layers[i][layer_index][1] for i in indices]
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
            estimator._band_groupable_layers()[band_index][layer_index][1]
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

        first_signatures = first._band_layer_signatures()
        if first_signatures is None:
            return False

        for estimator in estimators[1:]:
            if type(estimator) is not type(first):
                return False
            if estimator.training or not estimator.use_grouped_forward:
                return False
            if estimator.dim_inputs != first.dim_inputs:
                return False

            if estimator._band_layer_signatures() != first_signatures:
                return False
        return True

    def _forward_grouped_mlp(self, x):
        def forward_group(start, end):
            group_x = x[:, :, start:end, :]
            for kind, weight, bias in self._get_group_params(start, end, x.device, x.dtype):
                if kind == 'linear':
                    group_x = grouped_linear(group_x, weight, bias)
                else:
                    group_x = inference_tanh(group_x)
            return F.glu(group_x, dim=-1).flatten(start_dim=-2)

        return torch.cat([forward_group(start, end) for start, end, _ in self._dim_groups], dim=-1)

    def _forward_layer_grouped_mlp(self, x):
        plan = self._layer_grouping_plan()
        if plan is None:
            return None
        streamed = self._forward_two_layer_stream(x, plan)
        if streamed is not None:
            return streamed

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
                    selected = self._select_group(group_x, indices)
                    out = F.glu(grouped_linear(selected, weight, bias), dim=-1)
                    for band_position, band_out in zip(indices, out.unbind(dim=-2)):
                        outs[band_position] = band_out
                return torch.cat(outs, dim=-1)

            next_x = group_x.new_empty(*group_x.shape[:-1], next(iter(out_dims)))
            for signature, indices in groups:
                weight, bias = self._get_layer_group_params(layer_index, signature, indices, x.device, x.dtype)
                self._copy_group_(next_x, indices, grouped_linear(self._select_group(group_x, indices), weight, bias))
            group_x = next_x

        return F.glu(group_x, dim=-1).flatten(start_dim=-2)

    def _forward_two_layer_stream(self, x, plan):
        if len(plan) != 3 or plan[0][0] != 'linear' or plan[1][0] != 'tanh' or plan[2][0] != 'linear' or len(plan[0][1]) != 1:
            return None
        if torch.is_grad_enabled() or not x.is_cuda or x.dtype != torch.float16 or not triton_kernels_available():
            return None

        first_signature, _ = plan[0][1][0]
        result = x.new_empty(x.shape[0], x.shape[1], sum(self.dim_inputs))

        for final_signature, indices in plan[2][1]:
            weight, bias = self._get_layer_group_params(0, first_signature, indices, x.device, x.dtype)
            group_x = grouped_linear(self._select_group(x, indices), weight, bias)
            weight, bias = self._get_layer_group_params(2, final_signature, indices, x.device, x.dtype)
            out_dim = final_signature[1] // 2
            bounds = self._contiguous_bounds(indices)
            offset_start = None if bounds is None else self._dim_offsets[bounds[0]]

            if (
                bounds is not None
                and self._dim_offsets[bounds[1]] - offset_start == len(indices) * out_dim
                and mask_final_tanh_glu_flat_triton(group_x, weight, bias, result, offset_start, out_dim)
            ):
                continue

            group_x = F.glu(grouped_linear(inference_tanh(group_x), weight, bias), dim=-1)
            if bounds is not None:
                result[:, :, offset_start:self._dim_offsets[bounds[1]]] = group_x.flatten(start_dim=-2)
            else:
                for group_position, band_position in enumerate(indices):
                    offset_start = self._dim_offsets[band_position]
                    offset_end = self._dim_offsets[band_position + 1]
                    result[:, :, offset_start:offset_end] = group_x[:, :, group_position, :]

        return result

    def _forward_by_band_fast(self, x):
        band_layers = self._band_groupable_layers()

        def forward_band(band_index, band_features):
            group_x = band_features
            layers = band_layers[band_index]
            if layers is None:
                return None

            for kind, layer in layers:
                if kind == 'tanh':
                    group_x = inference_tanh(group_x)
                else:
                    group_x = layer(group_x)
            return F.glu(group_x, dim=-1)

        outs = [forward_band(band_index, band_features) for band_index, band_features in enumerate(x.unbind(dim=-2))]
        return None if any(out is None for out in outs) else torch.cat(outs, dim=-1)

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

    def _select_packed_group(self, group_x, indices, stem_count):
        selected = self._select_group(group_x, indices)
        if group_x.ndim == 4:
            return selected.unsqueeze(2).expand(-1, -1, stem_count, -1, -1)
        return selected

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
            group_x = first._select_group(x, indices).unsqueeze(2).expand(-1, -1, stem_count, -1, -1)
            b, t, s, g, d = group_x.shape
            group_x = grouped_linear(group_x.reshape(b, t, s * g, d), weight, bias)

            weight, bias = first._get_packed_layer_group_params(
                estimators, 2, final_signature, indices, x.device, x.dtype
            )
            out_dim = final_signature[1] // 2
            bounds = first._contiguous_bounds(indices)
            offset_start = None if bounds is None else first._dim_offsets[bounds[0]]
            if (
                bounds is not None
                and first._dim_offsets[bounds[1]] - offset_start == len(indices) * out_dim
                and mask_final_tanh_glu_packed_triton(group_x, weight, bias, result, len(indices), offset_start, out_dim)
            ):
                continue

            group_x = F.glu(grouped_linear(inference_tanh(group_x), weight, bias), dim=-1).reshape(b, t, s, g, -1)
            if bounds is not None:
                result[:, :, :, offset_start:first._dim_offsets[bounds[1]]] = group_x.flatten(start_dim=-2).transpose(1, 2)
            else:
                for group_position, band_position in enumerate(indices):
                    offset_start = first._dim_offsets[band_position]
                    offset_end = first._dim_offsets[band_position + 1]
                    result[:, :, :, offset_start:offset_end] = group_x[:, :, :, group_position, :].transpose(1, 2)

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
                    selected = first._select_packed_group(group_x, indices, stem_count)
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
                selected = first._select_packed_group(group_x, indices, stem_count)
                b, t, s, g, d = selected.shape
                out = grouped_linear(selected.reshape(b, t, s * g, d), weight, bias)
                first._copy_group_(next_x, indices, out.reshape(b, t, s, g, out_dim))
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
        result = x.new_empty(x.shape[0], stem_count, x.shape[1], first._dim_total)
        band_layers = first._band_groupable_layers()

        for band_index, layers in enumerate(band_layers):
            group_x = x[:, :, band_index, :].unsqueeze(-2).expand(-1, -1, stem_count, -1)
            for layer_index, (kind, layer) in enumerate(layers):
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
            result[:, :, :, offset_start:offset_end] = F.glu(group_x, dim=-1).transpose(1, 2)

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

        return torch.cat([mlp(band_features) for band_features, mlp in zip(x.unbind(dim=-2), self.to_freqs)], dim=-1)

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

    @staticmethod
    def warm_packed_estimators(estimators, device, dtype):
        estimators = tuple(estimators)
        if not estimators:
            return
        if not MaskEstimator._packable_estimators(estimators):
            MaskEstimator._warm_packed_estimators_by_band(estimators, device, dtype)
            return

        first = estimators[0]
        plan = first._layer_grouping_plan()
        if len(plan) == 3 and plan[0][0] == 'linear' and plan[1][0] == 'tanh' and plan[2][0] == 'linear' and len(plan[0][1]) == 1:
            first_signature, _ = plan[0][1][0]
            for final_signature, indices in plan[2][1]:
                first._get_packed_layer_group_params(estimators, 0, first_signature, indices, device, dtype)
                first._get_packed_layer_group_params(estimators, 2, final_signature, indices, device, dtype)
            return

        for layer_index, (kind, groups) in enumerate(plan):
            if kind == 'tanh':
                continue
            for signature, indices in groups:
                first._get_packed_layer_group_params(estimators, layer_index, signature, indices, device, dtype)

    @staticmethod
    def _warm_packed_estimators_by_band(estimators, device, dtype):
        if not MaskEstimator._packable_estimators_by_band(estimators):
            return
        first = estimators[0]
        for band_index, layers in enumerate(first._band_groupable_layers()):
            for layer_index, (kind, layer) in enumerate(layers):
                if kind != 'tanh':
                    first._get_packed_layer_group_params(estimators, layer_index, (layer.in_features, layer.out_features, layer.bias is not None), (band_index,), device, dtype)
