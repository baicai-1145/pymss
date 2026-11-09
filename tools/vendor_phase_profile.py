"""Phase-attributed GPU timing for the current Biren bf16+FMHA baseline.

Synchronize-bracketed regions attribute wall time to STFT, band_split, each of
the 12 (time, freq) transformer pairs, final_norm, mask estimation and ISTFT.
Attention internals (rotary, native FMHA) are timed by wrapping the exact
functions the model calls. Serialized attribution totals exceed the pipelined
forward by design; report both and never use this probe as a performance score.
"""
from __future__ import annotations
import argparse
import collections
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', default='.tmp/br-phase-profile.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    import torch
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    import pymss_core.modules.bs_roformer.common as common
    import pymss_core.modules.bs_roformer.transformer as transformer
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    from pymss_core.vendor_shims import prepare_model_for_vendor_rope

    records = collections.defaultdict(list)
    depth = 12

    def timed(fn, label):
        def wrapper(*call_args, **kwargs):
            torch.supa.synchronize()
            start = time.perf_counter()
            try:
                return fn(*call_args, **kwargs)
            finally:
                torch.supa.synchronize()
                records[label].append((time.perf_counter() - start) * 1000)
        return wrapper

    torch.manual_seed(0)
    model = BSRoformer(dim=384, depth=depth, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0').to(torch.bfloat16)
    prepare_model_for_vendor_rope(model)
    x = torch.randn(1, 2, int(44100 * args.seconds), device='supa:0') * 0.1
    with torch.no_grad():
        for _ in range(3):
            model(x)
        torch.supa.synchronize()
        plain = []
        for _ in range(2):
            start = time.perf_counter()
            model(x)
            torch.supa.synchronize()
            plain.append((time.perf_counter() - start) * 1000)
        print('PHASE_PIPELINED_MS', round(statistics.median(plain), 1), flush=True)

    originals = (common.stft_roformer, common.istft_roformer,
                 transformer.rotate_qk_fast_bnhd,
                 torch.ops.supa.sudnn_scaled_dot_product_attention_fwd,
                 BSRoformer._estimate_masks)

    original_rotary = transformer.rotate_qk_fast_bnhd
    original_fmha = torch.ops.supa.sudnn_scaled_dot_product_attention_fwd

    def side_of(batch):
        return 'freq' if batch > 1024 else 'time'

    common.stft_roformer = timed(common.stft_roformer, 'stft')
    common.istft_roformer = timed(common.istft_roformer, 'istft')

    def timed_rotary(rotary_embed, q, k):
        side = side_of(q.shape[0])
        torch.supa.synchronize()
        start = time.perf_counter()
        try:
            return transformer.rotate_qk_fast_bnhd(rotary_embed, q, k)
        finally:
            torch.supa.synchronize()
            records['attention.rotary.' + side].append((time.perf_counter() - start) * 1000)

    def timed_fmha(q, k, v, *call_args, **kwargs):
        side = side_of(q.reshape(-1, q.shape[-2], q.shape[-1]).shape[0])
        torch.supa.synchronize()
        start = time.perf_counter()
        try:
            return original_fmha(q, k, v, *call_args, **kwargs)
        finally:
            torch.supa.synchronize()
            records['attention.fmha.' + side].append((time.perf_counter() - start) * 1000)

    transformer.rotate_qk_fast_bnhd = timed_rotary
    torch.ops.supa.sudnn_scaled_dot_product_attention_fwd = timed_fmha
    BSRoformer._estimate_masks = timed(BSRoformer._estimate_masks, 'mask_estimate')

    handles = []
    hooks = [('band_split', model.band_split), ('final_norm', model.final_norm)]
    for index in range(depth):
        pair = model.layers[index]
        hooks.append((f'layer{index:02d}.time', pair[0]))
        hooks.append((f'layer{index:02d}.freq', pair[1]))
        if index == 0:
            for tag, transformer_module in (('t', pair[0]), ('f', pair[1])):
                attention = transformer_module.layers[0][0]
                feedforward = transformer_module.layers[0][1]
                hooks.append((f'layer00{tag}.attn', attention))
                hooks.append((f'layer00{tag}.ffn', feedforward))
                hooks.append((f'layer00{tag}.attn.qkv', attention.to_qkv))
                hooks.append((f'layer00{tag}.attn.out', attention.to_out))
                hooks.append((f'layer00{tag}.ffn.net', feedforward.net))

    def attach(label, module):
        def before(mod, inputs):
            torch.supa.synchronize()
            module._phase_start = time.perf_counter()
        def after(mod, inputs, output):
            torch.supa.synchronize()
            records[label].append((time.perf_counter() - module._phase_start) * 1000)
        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after, always_call=True))

    try:
        for label, module in hooks:
            attach(label, module)
        with torch.no_grad():
            for _ in range(args.repeat):
                model(x)
            torch.supa.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        (common.stft_roformer, common.istft_roformer, transformer.rotate_qk_fast_bnhd,
         torch.ops.supa.sudnn_scaled_dot_product_attention_fwd, BSRoformer._estimate_masks) = originals

    summary = {}
    for label, values in records.items():
        summary[label] = {'median_ms': round(statistics.median(values), 1),
                          'min_ms': round(min(values), 1), 'count': len(values)}
    def is_top(label):
        if label in ('stft', 'istft', 'band_split', 'final_norm', 'mask_estimate'):
            return True
        parts = label.split('.')
        return len(parts) == 2 and parts[0].startswith('layer') and parts[1] in ('time', 'freq')

    aggregate = collections.Counter()
    for label, entry in summary.items():
        parts = label.split('.')
        group = 'layer.' + parts[1] if parts[0].startswith('layer') else parts[0]
        aggregate[group] += entry['median_ms']
    serialized_total = round(sum(summary[label]['median_ms'] for label in summary if is_top(label)), 1)
    report = {'pipelined_median_ms': round(statistics.median(plain), 1),
              'serialized_top_sum_ms': serialized_total,
              'phases': dict(sorted(summary.items(), key=lambda item: -item[1]['median_ms'])),
              'aggregates': dict(aggregate.most_common())}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print('PHASE_SERIALIZED_SUM_MS', serialized_total, flush=True)
    for label, total in aggregate.most_common():
        print(f'PHASE_GROUP {label} {round(total, 1)}', flush=True)
    for label, entry in summary.items():
        if entry['median_ms'] >= 5 or label.startswith(('attention.', 'layer00')):
            print(f"PHASE {label} {entry['median_ms']} (n={entry['count']})", flush=True)
    print('PHASE_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
