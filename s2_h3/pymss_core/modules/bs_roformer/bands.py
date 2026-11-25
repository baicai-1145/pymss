import os
from collections import defaultdict
from itertools import accumulate, pairwise
import torch
from pymss_core.runtime_cache import _clear_cache_dicts
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList
from .transformer import RMSNorm
EXPERIMENTAL_TRAIN_GROUPED_BANDS_ENV = "PYMSS_CORE_EXPERIMENTAL_TRAIN_GROUPED_BANDS"
EXPERIMENTAL_DEEP_MASK_GROUPING_ENV = "PYMSS_CORE_EXPERIMENTAL_DEEP_MASK_GROUPING"
def _env_flag(name): return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}
experimental_train_grouped_bands = lambda: _env_flag(EXPERIMENTAL_TRAIN_GROUPED_BANDS_ENV)
experimental_deep_mask_grouping = lambda: _env_flag(EXPERIMENTAL_DEEP_MASK_GROUPING_ENV)
def should_use_grouped_forward(module): return module.use_grouped_forward and (not module.training or experimental_train_grouped_bands())
def default(v, d): return v if v is not None else d
def dim_input_offsets(dim_inputs): return (0, *accumulate(dim_inputs))
def contiguous_dim_groups(dim_inputs): breaks = [0] + [i for i in range(1, len(dim_inputs)) if dim_inputs[i] != dim_inputs[i - 1]] + [len(dim_inputs)]; return tuple((s, e, dim_inputs[s]) for s, e in pairwise(breaks))
def grouped_linear(x, weight, bias):
    group_count, out_features, in_features = weight.shape
    leading_shape = x.shape[:-2]
    x = x.reshape(-1, group_count, in_features).transpose(0, 1)
    w = weight.transpose(1, 2)
    if bias is not None and (bias.dtype != x.dtype or bias.device != x.device): bias = bias.to(device=x.device, dtype=x.dtype)
    try:
        if bias is None:
            out = torch.bmm(x, w)
        else:
            out = torch.baddbmm(bias.unsqueeze(1), x, w)
    except RuntimeError as error:
        if x.device.type != "supa" or ("Sublas" not in str(error) and "no valid backend" not in str(error)):
            raise
        # batched GEMM shape limits (SUBLAS ErrorCode 7): per-group 2-D matmul
        # covers the same algebra with shapes the 2-D kernels accept, then CPU
        try:
            outs = []
            for g in range(group_count):
                y = torch.matmul(x[g], w[g])
                if bias is not None: y = y + bias[g]
                outs.append(y)
            out = torch.stack(outs, dim=0)
        except RuntimeError:
            cpu_x, cpu_w = x.cpu(), w.cpu()
            outs = []
            for g in range(group_count):
                y = torch.matmul(cpu_x[g], cpu_w[g])
                if bias is not None: y = y + bias[g].cpu()
                outs.append(y)
            out = torch.stack(outs, dim=0).to(x.device)
    return out.transpose(0, 1).reshape(*leading_shape, group_count, out_features)
def inference_tanh(x): return torch.tanh(x) if torch.is_grad_enabled() else x.tanh_()
def stack_linears(linears, device, dtype): weight = torch.stack([linear.weight.to(device=device, dtype=dtype) for linear in linears], dim=0); bias = None if linears[0].bias is None else torch.stack([linear.bias.to(device=device, dtype=dtype) for linear in linears]); return weight, bias
class BandSplit(Module):
    def __init__(self, dim, dim_inputs: tuple[int, ...]): super().__init__(); self.dim_inputs, self._dim_offsets = dim_inputs, dim_input_offsets(dim_inputs); self._dim_groups, self._group_cache = contiguous_dim_groups(dim_inputs), {}; self.use_grouped_forward = True; self.to_features = ModuleList([nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim)) for dim_in in dim_inputs])
    def clear_runtime_cache(self): _clear_cache_dicts(self, "_group_cache")
    def _get_group_params(self, start, end, device, dtype):
        key = (start, end, device.type, device.index, dtype)
        use_cache = not self.training
        if use_cache:
            cached = self._group_cache.get(key)
            if cached is not None: return cached
        norms = [self.to_features[i][0] for i in range(start, end)]
        linears = [self.to_features[i][1] for i in range(start, end)]
        gamma = torch.stack([norm.gamma.to(device=device, dtype=dtype) for norm in norms], dim=0) * (self.dim_inputs[start] ** 0.5)
        weight, bias = stack_linears(linears, device, dtype)
        cached = (gamma, weight, bias)
        if use_cache: self._group_cache[key] = cached
        return cached
    def _forward_grouped(self, x):
        def forward_group(start, end, dim_in): offset_start = self._dim_offsets[start]; offset_end = self._dim_offsets[end]; group_x = x[..., offset_start:offset_end].reshape(*x.shape[:-1], end - start, dim_in); gamma, weight, bias = self._get_group_params(start, end, x.device, x.dtype); group_x = F.normalize(group_x, dim=-1) * gamma; return grouped_linear(group_x, weight, bias)
        return torch.cat([forward_group(start, end, dim_in) for start, end, dim_in in self._dim_groups], dim=-2)
    def warm_group_cache(self, device, dtype):
        if self.training: return
        for start, end, _ in self._dim_groups: self._get_group_params(start, end, device, dtype)
    def forward(self, x): return self._forward_grouped(x) if (should_use_grouped_forward(self)) else torch.stack([to_feature(split_input) for split_input, to_feature in zip(x.split(self.dim_inputs, dim=-1), self.to_features)], dim=-2)
def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh, hidden_layers=None): dim_hidden, hidden_layers = default(dim_hidden, dim_in), default(hidden_layers, max(depth - 1, 0)); dims = (dim_in, *((dim_hidden,) * hidden_layers), dim_out); return nn.Sequential(*[layer for ind, (i, o) in enumerate(pairwise(dims)) for layer in ((nn.Linear(i, o),) if ind == len(dims) - 2 else (nn.Linear(i, o), activation()))])
class MaskEstimator(Module):
    def __init__(self, dim, dim_inputs: tuple[int, ...], depth, mlp_expansion_factor=4, mlp_hidden_layers=None):
        super().__init__()
        self.dim_inputs = dim_inputs
        self._dim_total = sum(dim_inputs)
        self._dim_offsets = dim_input_offsets(dim_inputs)
        self._dim_groups = contiguous_dim_groups(dim_inputs)
        self._group_cache, self._layer_group_cache, self._index_cache, self._packed_layer_group_cache = {}, {}, {}, {}
        self._band_layers_cache = self._band_signatures_cache = self._layer_group_plan = self._can_group_mlp_cache = None
        self._layer_group_plan_ready = False
        self.use_grouped_forward = True
        # _groupable_layers_cache intentionally removed: id()-keyed caching is
        # unsafe under DataParallel replica recycling (see _groupable_layers).
        dim_hidden = dim * mlp_expansion_factor
        self.to_freqs = ModuleList([ nn.Sequential(MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth, hidden_layers=mlp_hidden_layers), nn.GLU(dim=-1)) for dim_in in dim_inputs ])
    def clear_runtime_cache(self):
        _clear_cache_dicts(self, "_group_cache", "_layer_group_cache", "_index_cache", "_packed_layer_group_cache")
    def _groupable_layers(self, mlp_with_glu):
        # NOTE: Do not cache by id(mlp_with_glu). Under torch.nn.DataParallel the
        # module is re-replicated on every forward, so band submodules are short
        # lived and Python recycles their id() values. A shared id-keyed cache
        # then returns another band's parsed layers, corrupting per-band output
        # widths. The per-replica memo in _band_groupable_layers is enough.
        if not isinstance(mlp_with_glu, nn.Sequential) or len(mlp_with_glu) != 2: return None
        mlp, glu = mlp_with_glu
        if not isinstance(glu, nn.GLU) or not isinstance(mlp, nn.Sequential): return None
        layers = [(("linear", layer) if isinstance(layer, nn.Linear) else ("tanh", None) if isinstance(layer, nn.Tanh) else None) for layer in mlp]
        if not layers or any(layer is None for layer in layers) or layers[-1][0] != "linear": return None
        return tuple(layers)
    def _band_groupable_layers(self):
        if self._band_layers_cache is None: self._band_layers_cache = tuple((self._groupable_layers(mlp_with_glu) for mlp_with_glu in self.to_freqs))
        return self._band_layers_cache
    @staticmethod
    def _layers_signature(layers): return tuple(item if kind != "linear" else (kind, item.in_features, item.out_features, item.bias is not None) for kind, item in layers)
    def _band_layer_signatures(self):
        if self._band_signatures_cache is None: layers = self._band_groupable_layers(); self._band_signatures_cache = None if any((layer_group is None for layer_group in layers)) else tuple((self._layers_signature(layer_group) for layer_group in layers))
        return self._band_signatures_cache
    def _plan_bail(self, allow_deep_grouping): self._layer_group_plan_ready,self._layer_group_plan_allow_deep,self._layer_group_plan = True, allow_deep_grouping, None
    def _layer_grouping_plan(self):
        allow_deep_grouping = self.training and experimental_deep_mask_grouping()
        if self._layer_group_plan_ready and getattr(self, "_layer_group_plan_allow_deep", False) == allow_deep_grouping: return self._layer_group_plan
        band_layers = self._band_groupable_layers()
        if any(layers is None for layers in band_layers): return self._plan_bail(allow_deep_grouping)
        layer_count = len(band_layers[0])
        if any(len(layers) != layer_count for layers in band_layers): return self._plan_bail(allow_deep_grouping)
        plan = []
        for layer_index in range(layer_count):
            kinds = [layers[layer_index][0] for layers in band_layers]
            if kinds[0] == "tanh":
                if any(kind != "tanh" for kind in kinds): return self._plan_bail(allow_deep_grouping)
                plan.append(("tanh", None))
                continue
            if kinds[0] != "linear" or any(kind != "linear" for kind in kinds): return self._plan_bail(allow_deep_grouping)
            groups = defaultdict(list)
            for band_index, layers in enumerate(band_layers): _kind, layer = layers[layer_index]; groups[(layer.in_features, layer.out_features, layer.bias is not None)].append(band_index)
            plan.append(("linear", tuple((signature, tuple(indices)) for signature, indices in groups.items())))
        # Extra hidden layers in MBR produce large per-band hidden->hidden
        # batched GEMMs. On CUDA those are slower than the existing addmm
        # loop, so keep this fast path to the common two-linear mask heads.
        if sum(1 for kind, _ in plan if kind == "linear") > 2 and not allow_deep_grouping: return self._plan_bail(allow_deep_grouping)
        self._layer_group_plan_ready = True
        self._layer_group_plan_allow_deep = allow_deep_grouping
        self._layer_group_plan = tuple(plan)
        return self._layer_group_plan
    def _can_group_mlp(self):
        if self._can_group_mlp_cache is not None: return self._can_group_mlp_cache
        signatures = self._band_layer_signatures()
        self._can_group_mlp_cache = signatures is not None and all(signature == signatures[0] for signature in signatures)
        return self._can_group_mlp_cache
    def _indices_tensor(self, indices, device):
        key = (indices, device.type, device.index)
        cached = self._index_cache.get(key)
        if cached is None or cached.device != device: cached = torch.as_tensor(indices, device=device); self._index_cache[key] = cached
        return cached
    def _get_group_params(self, start, end, device, dtype):
        key = (start, end, device.type, device.index, dtype)
        use_cache = not self.training
        if use_cache:
            cached = self._group_cache.get(key)
            if cached is not None: return cached
        grouped_layers, band_layers = [], self._band_groupable_layers()
        first_layers = band_layers[start]
        for layer_index, (kind, _) in enumerate(first_layers):
            if kind == "tanh": grouped_layers.append(('tanh', None, None)); continue
            weight, bias = stack_linears([band_layers[i][layer_index][1] for i in range(start, end)], device, dtype)
            grouped_layers.append(("linear", weight, bias))
        cached = tuple(grouped_layers)
        if use_cache: self._group_cache[key] = cached
        return cached
    def _get_layer_group_params(self, layer_index, signature, indices, device, dtype):
        key = (layer_index, signature, indices, device.type, device.index, dtype)
        use_cache = not self.training
        if use_cache:
            cached = self._layer_group_cache.get(key)
            if cached is not None: return cached
        band_layers = self._band_groupable_layers()
        weight, bias = stack_linears([band_layers[i][layer_index][1] for i in indices], device, dtype)
        cached = (weight, bias)
        if use_cache: self._layer_group_cache[key] = cached
        return cached
    def _get_packed_layer_group_params(self, estimators, layer_index, signature, indices, device, dtype):
        estimator_ids = tuple(id(estimator) for estimator in estimators)
        key = (estimator_ids, layer_index, signature, indices, device.type, device.index, dtype)
        use_cache = not any(estimator.training for estimator in estimators)
        if use_cache:
            cached = self._packed_layer_group_cache.get(key)
            if cached is not None: return cached
        linears = [ estimator._band_groupable_layers()[band_index][layer_index][1] for estimator in estimators for band_index in indices ]
        weight, bias = stack_linears(linears, device, dtype)
        cached = (weight, bias)
        if use_cache: self._packed_layer_group_cache[key] = cached
        return cached
    @staticmethod
    def _checked_estimators(estimators):
        estimators = tuple(estimators)
        if len(estimators) <= 1 or not isinstance(estimators[0], MaskEstimator) or not should_use_grouped_forward(estimators[0]): return None
        return estimators
    @staticmethod
    def _packable_estimators_by_band(estimators):
        estimators = MaskEstimator._checked_estimators(estimators)
        if estimators is None: return False
        first, first_signatures = estimators[0], estimators[0]._band_layer_signatures()
        if first_signatures is None: return False
        return all(type(e) is type(first) and should_use_grouped_forward(e) and e.dim_inputs == first.dim_inputs and e._band_layer_signatures() == first_signatures for e in estimators[1:])
    def _forward_grouped_mlp(self, x):
        def forward_group(start, end):
            group_x = x[:, :, start:end, :]
            for kind, weight, bias in self._get_group_params(start, end, x.device, x.dtype): group_x = grouped_linear(group_x, weight, bias) if kind == "linear" else inference_tanh(group_x)
            return F.glu(group_x, dim=-1).flatten(start_dim=-2)
        return torch.cat([forward_group(start, end) for start, end, _ in self._dim_groups], dim=-1)
    def _merge_layer_groups(self, outs, groups, device):
        result = torch.cat(outs, dim=-2) if len(outs) > 1 else outs[0]
        order = tuple(index for _, indices in groups for index in indices)
        if order != tuple(range(len(order))):
            inverse = tuple(sorted(range(len(order)), key=order.__getitem__))
            result = result.index_select(-2, self._indices_tensor(inverse, device))
        return result
    def _forward_layer_grouped_mlp(self, x):
        plan = self._layer_grouping_plan()
        if plan is None: return None
        group_x = x
        for layer_index, (kind, groups) in enumerate(plan):
            if kind == "tanh": group_x = inference_tanh(group_x); continue
            out_dims = {signature[1] for signature, _ in groups}
            if len(out_dims) != 1:
                if layer_index != len(plan) - 1: return None
                outs = [None] * len(self.to_freqs)
                for signature, indices in groups:
                    weight, bias = self._get_layer_group_params(layer_index, signature, indices, x.device, x.dtype)
                    band_index = self._indices_tensor(indices, x.device)
                    selected = group_x.index_select(-2, band_index)
                    out = F.glu(grouped_linear(selected, weight, bias), dim=-1)
                    for band_position, band_out in zip(indices, out.unbind(dim=-2)): outs[band_position] = band_out
                return torch.cat(outs, dim=-1)
            outs = []
            for signature, indices in groups:
                weight, bias = self._get_layer_group_params(layer_index, signature, indices, x.device, x.dtype)
                band_index = self._indices_tensor(indices, x.device)
                selected = group_x.index_select(-2, band_index)
                out = grouped_linear(selected, weight, bias)
                # Avoid index_copy_ for AscendIR, preserving noncontiguous band groups.
                outs.append(out)
            group_x = self._merge_layer_groups(outs, groups, x.device)
        return F.glu(group_x, dim=-1).flatten(start_dim=-2)
    def _forward_by_band_fast(self, x):
        band_layers = self._band_groupable_layers()
        def forward_band(band_index, band_features):
            group_x = band_features
            layers = band_layers[band_index]
            if layers is None: return None
            for kind, layer in layers: group_x = inference_tanh(group_x) if kind == "tanh" else layer(group_x)
            return F.glu(group_x, dim=-1)
        outs = [forward_band(band_index, band_features) for band_index, band_features in enumerate(x.unbind(dim=-2))]
        return None if any(out is None for out in outs) else torch.cat(outs, dim=-1)
    @staticmethod
    def _packable_estimators(estimators):
        estimators = MaskEstimator._checked_estimators(estimators)
        if estimators is None: return False
        first = estimators[0]
        first_plan = first._layer_grouping_plan()
        if first_plan is None: return False
        return all(type(estimator) is type(first) and should_use_grouped_forward(estimator) and estimator.dim_inputs == first.dim_inputs and estimator._layer_grouping_plan() == first_plan for estimator in estimators[1:])
    @staticmethod
    def _select_packed_group(group_x, band_index, stem_count):
        if group_x.ndim == 4: return group_x.index_select(-2, band_index).unsqueeze(2).expand(-1, -1, stem_count, -1, -1)
        return group_x.index_select(-2, band_index)
    @staticmethod
    def _forward_packed_estimators_two_layer_stream(estimators, x, plan):
        if len(plan) != 3 or plan[0][0] != "linear" or plan[1][0] != "tanh" or plan[2][0] != "linear": return None
        first,first_groups,final_groups = estimators[0], plan[0][1], plan[2][1]
        if len(first_groups) != 1: return None
        first_signature, _ = first_groups[0]
        stem_count = len(estimators)
        result = x.new_empty(x.shape[0], stem_count, x.shape[1], sum(first.dim_inputs))
        for final_signature, indices in final_groups:
            weight, bias = first._get_packed_layer_group_params(estimators, 0, first_signature, indices, x.device, x.dtype)
            band_index = first._indices_tensor(indices, x.device)
            group_x = x.index_select(-2, band_index).unsqueeze(2).expand(-1, -1, stem_count, -1, -1)
            b, t, s, g, d = group_x.shape
            group_x = grouped_linear(group_x.reshape(b, t, s * g, d), weight, bias)
            group_x = inference_tanh(group_x)
            weight, bias = first._get_packed_layer_group_params(estimators, 2, final_signature, indices, x.device, x.dtype)
            group_x = grouped_linear(group_x, weight, bias)
            group_x = F.glu(group_x, dim=-1).reshape(b, t, s, g, -1)
            if indices == tuple(range(indices[0], indices[-1] + 1)):
                offset_start = first._dim_offsets[indices[0]]
                offset_end = first._dim_offsets[indices[-1] + 1]
                result[:, :, :, offset_start:offset_end] = group_x.flatten(start_dim=-2).transpose(1, 2)
            else:
                for group_position, band_position in enumerate(indices): offset_start = first._dim_offsets[band_position]; offset_end = first._dim_offsets[band_position + 1]; result[:, :, :, offset_start:offset_end] = group_x[:, :, :, group_position, :].transpose(1, 2)
        return result
    @staticmethod
    def forward_packed_estimators(estimators, x):
        estimators = tuple(estimators)
        if not MaskEstimator._packable_estimators(estimators): return MaskEstimator._forward_packed_estimators_by_band(estimators, x)
        first = estimators[0]
        plan = first._layer_grouping_plan()
        streamed = MaskEstimator._forward_packed_estimators_two_layer_stream(estimators, x, plan)
        if streamed is not None: return streamed
        stem_count,group_x = len(estimators), x
        for layer_index, (kind, groups) in enumerate(plan):
            if kind == "tanh": group_x = inference_tanh(group_x); continue
            out_dims = {signature[1] for signature, _ in groups}
            if len(out_dims) != 1:
                if layer_index != len(plan) - 1: return None
                result = x.new_empty(x.shape[0], x.shape[1], stem_count, sum(first.dim_inputs))
                for signature, indices in groups:
                    weight, bias = first._get_packed_layer_group_params(estimators, layer_index, signature, indices, x.device, x.dtype)
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
                        for group_position, band_position in enumerate(indices): offset_start = first._dim_offsets[band_position]; offset_end = first._dim_offsets[band_position + 1]; result[:, :, :, offset_start:offset_end] = out[:, :, :, group_position, :]
                return result.permute(0, 2, 1, 3)
            out_dim = next(iter(out_dims))
            outs = []
            for signature, indices in groups:
                weight, bias = first._get_packed_layer_group_params(estimators, layer_index, signature, indices, x.device, x.dtype)
                band_index = first._indices_tensor(indices, x.device)
                selected = MaskEstimator._select_packed_group(group_x, band_index, stem_count)
                b, t, s, g, d = selected.shape
                out = grouped_linear(selected.reshape(b, t, s * g, d), weight, bias)
                out = out.reshape(b, t, s, g, out_dim)
                outs.append(out)
            group_x = first._merge_layer_groups(outs, groups, x.device)
        out = F.glu(group_x, dim=-1).flatten(start_dim=-2)
        return out.permute(0, 2, 1, 3)
    @staticmethod
    def _forward_packed_estimators_by_band(estimators, x):
        estimators = tuple(estimators)
        if not MaskEstimator._packable_estimators_by_band(estimators): return None
        first = estimators[0]
        stem_count = len(estimators)
        result = x.new_empty(x.shape[0], stem_count, x.shape[1], first._dim_total)
        band_layers = first._band_groupable_layers()
        for band_index, layers in enumerate(band_layers):
            group_x = x[:, :, band_index, :].unsqueeze(-2).expand(-1, -1, stem_count, -1)
            for layer_index, (kind, layer) in enumerate(layers):
                if kind == "tanh": group_x = inference_tanh(group_x); continue
                signature = (layer.in_features, layer.out_features, layer.bias is not None)
                weight, bias = first._get_packed_layer_group_params(estimators, layer_index, signature, (band_index,), x.device, x.dtype)
                group_x = grouped_linear(group_x, weight, bias)
            offset_start = first._dim_offsets[band_index]
            offset_end = first._dim_offsets[band_index + 1]
            result[:, :, :, offset_start:offset_end] = F.glu(group_x, dim=-1).transpose(1, 2)
        return result
    def forward(self, x):
        if should_use_grouped_forward(self):
            if self._can_group_mlp(): return self._forward_grouped_mlp(x)
            grouped = self._forward_layer_grouped_mlp(x)
            if grouped is not None: return grouped
            by_band = self._forward_by_band_fast(x)
            if by_band is not None: return by_band
        return torch.cat([mlp(band_features) for band_features, mlp in zip(x.unbind(dim=-2), self.to_freqs)], dim=-1)
    def warm_group_cache(self, device, dtype):
        if self.training: return
        if self._can_group_mlp():
            for start, end, _ in self._dim_groups: self._get_group_params(start, end, device, dtype)
            return
        plan = self._layer_grouping_plan()
        if plan is None: return
        for layer_index, (kind, groups) in enumerate(plan):
            if kind == "tanh": continue
            for signature, indices in groups: self._get_layer_group_params(layer_index, signature, indices, device, dtype)
    @staticmethod
    def warm_packed_estimators(estimators, device, dtype):
        estimators = tuple(estimators)
        if not estimators: return
        if any(estimator.training for estimator in estimators): return
        if not MaskEstimator._packable_estimators(estimators): MaskEstimator._warm_packed_estimators_by_band(estimators, device, dtype); return
        first = estimators[0]
        plan = first._layer_grouping_plan()
        if (len(plan) == 3 and plan[0][0] == "linear" and plan[1][0] == "tanh" and plan[2][0] == "linear" and len(plan[0][1]) == 1):
            first_signature, _ = plan[0][1][0]
            for final_signature, indices in plan[2][1]: first._get_packed_layer_group_params(estimators, 0, first_signature, indices, device, dtype); first._get_packed_layer_group_params(estimators, 2, final_signature, indices, device, dtype)
            return
        for layer_index, (kind, groups) in enumerate(plan):
            if kind == "tanh": continue
            for signature, indices in groups: first._get_packed_layer_group_params(estimators, layer_index, signature, indices, device, dtype)
    @staticmethod
    def _warm_packed_estimators_by_band(estimators, device, dtype):
        if not MaskEstimator._packable_estimators_by_band(estimators): return
        first = estimators[0]
        for band_index, layers in enumerate(first._band_groupable_layers()):
            for layer_index, (kind, layer) in enumerate(layers):
                if kind != "tanh": first._get_packed_layer_group_params(estimators, layer_index, (layer.in_features, layer.out_features, layer.bias is not None), (band_index,), device, dtype)
