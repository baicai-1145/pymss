"""Dissect the freq-side attention overhead bucket (regroup/rope/gates).

Fine-grained timing inside the installed SUPA fast path: norm, qkv projection,
the (3,B,H,S,D) regroup copy, rope (chunked vendor op), native FMHA, and the
gates+out tail -- recorded separately for time (batch<=1024) and freq inputs.
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
    parser.add_argument('--output', default='.tmp/br-freq-attn.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    import torch
    from pymss_core import vendor_shims
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    import pymss_core.modules.bs_roformer.transformer as transformer
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    from pymss_core.vendor_shims import _rope_chunked, prepare_model_for_vendor_rope

    records = collections.defaultdict(list)

    def stage(label, side):
        torch.supa.synchronize()
        start = time.perf_counter()
        return start, lambda: (records[label + '.' + side].append((time.perf_counter() - start) * 1000),
                               torch.supa.synchronize())

    def rope_bhqd_timed(rotary, q, k, side):
        b, h, n, d = q.shape
        start, done = stage('rope_tables', side)
        cos, sin = transformer.cached_rotary_cos_sin(rotary, n, q.device, torch.float32)
        cf = torch.cat((cos, cos), dim=-1).reshape(1, n, d).contiguous()
        sf = torch.cat((sin, sin), dim=-1).reshape(1, n, d).contiguous()
        done()
        outputs = []
        for tag, tensor in (('q', q), ('k', k)):
            start, done = stage('rope_flat_' + tag, side)
            flat = tensor.reshape(b * h, n, d)
            done()
            start, done = stage('rope_op_' + tag, side)
            out = _rope_chunked(flat, cf, sf, b * h)
            done()
            start, done = stage('rope_view_' + tag, side)
            out = out.view(b, h, n, d)
            done()
            outputs.append(out)
        return outputs[0], outputs[1]

    original_forward = transformer.Attention.forward

    def instrumented_forward(self, x):
        side = 'freq' if x.shape[0] > 1024 else 'time'
        if not (x.device.type == 'supa' and x.dtype == torch.bfloat16
                and getattr(self.rotary_embed, '_pymss_vendor_rope', False)):
            return original_forward(self, x)
        heads = self.heads
        dim_head = self.to_qkv.out_features // (3 * heads)
        start, done = stage('norm', side); normalized = self.norm(x); done()
        start, done = stage('qkv', side); qkv = self.to_qkv(normalized); done()
        b, n, _ = qkv.shape
        start, done = stage('regroup', side)
        trio = qkv.view(b, n, 3, heads, dim_head).permute(2, 0, 3, 1, 4).contiguous()
        done()
        start, done = stage('rope_total', side)
        q, k = rope_bhqd_timed(self.rotary_embed, trio[0], trio[1], side)
        done()
        start, done = stage('fmha', side); out = self._attention(q, k, trio[2]); done()
        start, done = stage('gates', side)
        gated = (out.transpose(1, 2) * self.to_gates(normalized).sigmoid().unsqueeze(-1)).flatten(start_dim=-2)
        done()
        start, done = stage('to_out', side); result = self.to_out(gated); done()
        return result

    torch.manual_seed(0)
    model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0').to(torch.bfloat16)
    assert len(prepare_model_for_vendor_rope(model)) == 24
    x = torch.randn(1, 2, int(44100 * args.seconds), device='supa:0') * 0.1
    transformer.Attention.forward = instrumented_forward
    try:
        with torch.no_grad():
            for _ in range(3):
                model(x)
            torch.supa.synchronize()
            for _ in range(args.repeat):
                model(x)
            torch.supa.synchronize()
    finally:
        transformer.Attention.forward = original_forward

    summary = {label: {'median_ms': round(statistics.median(values), 2), 'n': len(values)}
               for label, values in sorted(records.items())}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    for label, entry in summary.items():
        print(f"FREQ_ATTN {label} {entry['median_ms']} (n={entry['n']})", flush=True)
    print('FREQ_ATTN_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
