"""Validate SUPAGraph replay and compare host/wall times on Biren.

Each dtype and BRTB_ENABLE_GRAPH_EXECUTOR_ZERO_WS setting must run in a fresh
process. The environment switch is experimental, not assumed to be a speedup.
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    device = torch.device('supa:0')

    def measure(fn):
        torch.supa.synchronize()
        start = time.perf_counter()
        fn()
        host_ms = (time.perf_counter() - start) * 1000
        torch.supa.synchronize()
        wall_ms = (time.perf_counter() - start) * 1000
        return {'host_ms': host_ms, 'wall_ms': wall_ms}

    with torch.no_grad():
        # First prove that capture/replay works on a changed input buffer.
        tiny = torch.ones(1024, device=device)
        stream = torch.supa.Stream()
        stream.wait_stream(torch.supa.current_stream())
        with torch.supa.stream(stream):
            for _ in range(3):
                tiny + 1
        torch.supa.current_stream().wait_stream(stream)
        small_graph = torch.supa.SUPAGraph()
        with torch.supa.graph(small_graph, stream=stream):
            small_out = tiny + 1
        tiny.fill_(2)
        small_graph.replay()
        torch.supa.synchronize()
        torch.testing.assert_close(small_out, torch.full_like(tiny, 3))
        print('TINY_GRAPH', json.dumps({'changed_input_correct': True,
              'eager': measure(lambda: tiny + 1), 'replay': measure(small_graph.replay)}), flush=True)

        torch.manual_seed(0)
        model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                           time_transformer_depth=1, freq_transformer_depth=1).eval().to(device)
        if args.dtype == 'bf16':
            model.to(dtype=torch.bfloat16)
        static_x = torch.randn(1, 2, int(44100 * args.seconds), device=device) * 0.1
        stream.wait_stream(torch.supa.current_stream())
        with torch.supa.stream(stream):
            for _ in range(3):
                model(static_x)
        torch.supa.current_stream().wait_stream(stream)
        torch.supa.synchronize()
        print('MODEL_WARMED', args.dtype, flush=True)
        graph = torch.supa.SUPAGraph()
        with torch.supa.graph(graph, stream=stream):
            static_out = model(static_x)
        torch.supa.synchronize()
        graph.replay()
        torch.supa.synchronize()

        # AB/BA rounds: report enqueue time separately from synchronized latency.
        samples = {'eager': [], 'replay': []}
        for index in range(args.repeat):
            order = ('eager', 'replay') if index % 2 == 0 else ('replay', 'eager')
            for name in order:
                samples[name].append(measure(lambda: model(static_x)) if name == 'eager'
                                     else measure(graph.replay))
            print('GRAPH_PAIR', index, {name: samples[name][-1] for name in samples}, flush=True)

        captured_before = static_out.clone()
        static_x.mul_(0.7)
        expected = model(static_x).clone()
        graph.replay()
        torch.supa.synchronize()
        torch.testing.assert_close(static_out, expected, rtol=1e-4, atol=1e-5)
        changed = not torch.equal(captured_before, static_out)
        assert changed, 'replay returned the output for the old input'
        medians = {name: {metric: statistics.median(row[metric] for row in rows)
                         for metric in ('host_ms', 'wall_ms')}
                   for name, rows in samples.items()}
        print('GRAPH_RESULT', json.dumps({'dtype': args.dtype,
              'sdpa_backend': getattr(F.scaled_dot_product_attention, '_pymss_sdpa_backend', 'unpatched'),
              'zero_ws': os.environ.get('BRTB_ENABLE_GRAPH_EXECUTOR_ZERO_WS', 'unset'),
              'samples': samples, 'medians': medians, 'changed_input_correct': changed,
              'relative_l2': ((static_out - expected).norm() / expected.norm()).item()}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
