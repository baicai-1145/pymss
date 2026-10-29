"""Profile the full Biren bf16+FMHA path, including hidden host fallbacks.

CPU profiler timings diagnose dispatch/copies; GPU events in a separate forward
attribute module time without profiler overhead. Never use the profiled forward as
an end-to-end performance score. Preserve the inherited SDK PYTHONPATH.
"""
from __future__ import annotations
import argparse
import collections
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--output', default='.tmp/br-fmha-profile.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    import torch
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    torch.manual_seed(0)
    model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0').to(torch.bfloat16)
    x = torch.randn(1, 2, int(44100 * args.seconds), device='supa:0') * 0.1
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        torch.supa.synchronize()
        start = time.perf_counter()
        output = model(x)
        torch.supa.synchronize()
        baseline_ms = (time.perf_counter() - start) * 1000
        print('PROFILE_BASELINE_MS', baseline_ms, flush=True)

        contexts = collections.defaultdict(list)
        handles = []
        for name, module in model.named_modules():
            if list(module.children()):
                continue
            def before(mod, inputs, label=name):
                context = torch.profiler.record_function('module:' + label + ':' + type(mod).__name__)
                contexts[id(mod)].append(context)
                context.__enter__()
            def after(mod, inputs, output):
                contexts[id(mod)].pop().__exit__(None, None, None)
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after, always_call=True))
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                        record_shapes=True, with_stack=True) as prof:
                model(x)
                torch.supa.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        rows = sorted(prof.key_averages(), key=lambda row: row.self_cpu_time_total, reverse=True)
        cpu_top = [{'op': row.key, 'count': row.count,
                    'self_cpu_ms': row.self_cpu_time_total / 1000,
                    'total_cpu_ms': row.cpu_time_total / 1000}
                   for row in rows if not row.key.startswith('module:')][:25]
        copies = collections.Counter()
        for event in prof.events():
            if event.name != 'aten::_to_cpu':
                continue
            chain = []
            parent = event.cpu_parent
            for _ in range(12):
                if parent is None:
                    break
                chain.append(parent.name)
                parent = parent.cpu_parent
            copies[tuple(chain)] += 1
        copy_paths = [{'count': count, 'parents': list(chain)}
                      for chain, count in copies.most_common(15)]
        prof.export_chrome_trace(str(output_path.with_suffix('.trace.json')))
        print('PROFILE_CPU_TOP', json.dumps(cpu_top), flush=True)
        print('PROFILE_CPU_COPY_PATHS', json.dumps(copy_paths), flush=True)

        # Separate clean forward, with event pairs only around leaf modules.
        records = collections.defaultdict(list)
        pending = collections.defaultdict(list)
        handles = []
        for name, module in model.named_modules():
            if list(module.children()):
                continue
            def before(mod, inputs):
                event = torch.supa.Event(enable_timing=True)
                event.record()
                pending[id(mod)].append(event)
            def after(mod, inputs, output):
                end = torch.supa.Event(enable_timing=True)
                end.record()
                records[type(mod).__name__].append((pending[id(mod)].pop(), end))
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after))
        try:
            model(x)
            torch.supa.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        gpu_modules = [{'class': name, 'count': len(spans),
                        'gpu_ms': sum(start.elapsed_time(end) for start, end in spans)}
                       for name, spans in records.items()]
        gpu_modules.sort(key=lambda row: -row['gpu_ms'])
        print('PROFILE_GPU_LEAVES', json.dumps(gpu_modules), flush=True)
        summary = {'baseline_ms': baseline_ms, 'config': vars(args), 'cpu_top': cpu_top,
                   'cpu_copy_paths': copy_paths, 'gpu_leaf_modules': gpu_modules,
                   'output_finite': bool(output.isfinite().all())}
        output_path.write_text(json.dumps(summary, indent=2))
        print('PROFILE_ARTIFACT', str(output_path), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
