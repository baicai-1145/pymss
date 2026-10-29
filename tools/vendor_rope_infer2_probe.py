"""Contract + speed probe for supa_rope_infer_v2 (joint q/k RoPE op).

Signature observed in the SDK torch_br package (see vendor_fusion_contract_probe):
    out_q, ... = torch.ops.supa.supa_rope_infer_v2(q2d, k2d, sin, cos, position, dim, mode)[0]
with q2d/k2d shaped (batch*seq, heads*dim).  This probe establishes the exact
layout/dtype contract against a halves-rotation reference and times the op at
the model's real shapes, for comparison with the chunked supa_rope_apply path
(~15 GB/s effective there).
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch


def reference_rotation(x_bhqd, cos_half, sin_half):
    """x (B*H, S, D) interleaved pairs -> halves-rotation reference output."""
    even, odd = x_bhqd[..., ::2].float(), x_bhqd[..., 1::2].float()
    return torch.cat((even * cos_half - odd * sin_half,
                      odd * cos_half + even * sin_half), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.tmp/br-rope-infer2.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    device = torch.device('supa:0')
    op = getattr(torch.ops.supa, 'supa_rope_infer_v2', None)
    if op is None:
        print('INFER2_MISSING', flush=True)
        return 0
    results = []

    def attempt(tag, batch, seq, heads, dim, mode):
        x = (torch.randn(batch * heads, seq, dim, device=device) * 0.5).to(torch.bfloat16)
        q2d = x.reshape(batch * seq, heads * dim).contiguous()
        k2d = q2d.clone()
        positions = torch.arange(seq, device=device).repeat(batch).long()
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        angles = torch.arange(seq, device=device).float()[:, None] * inv[None, :]
        cos_half, sin_half = angles.cos(), angles.sin()          # (seq, dim/2)
        record = {'tag': tag, 'batch': batch, 'seq': seq, 'heads': heads, 'dim': dim, 'mode': mode}
        try:
            out = op(q2d, k2d, sin_half, cos_half, positions, dim, mode)
            q_out = out[0] if isinstance(out, (tuple, list)) else out
            torch.supa.synchronize()
            record['out_shape'] = list(q_out.shape)
            record['out_dtype'] = str(q_out.dtype)
            expected = reference_rotation(x, cos_half[None], sin_half[None])
            got = q_out.reshape(batch * heads, seq, dim).float()
            record['rel_l2'] = round(float((got - expected).norm() / expected.norm()), 6)
            record['finite'] = bool(got.isfinite().all())
            # warmed timing
            for _ in range(3):
                op(q2d, k2d, sin_half, cos_half, positions, dim, mode)
            torch.supa.synchronize()
            samples = []
            for _ in range(7):
                start = time.perf_counter()
                op(q2d, k2d, sin_half, cos_half, positions, dim, mode)
                torch.supa.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            record['median_ms'] = round(statistics.median(samples), 3)
        except Exception as error:
            record['error'] = f'{type(error).__name__}: ' + str(error).splitlines()[-1][:160]
        results.append(record)
        print('INFER2', json.dumps(record), flush=True)

    for mode in ('Half', 'Neox', 'half', 'neox'):
        attempt('small', 2, 17, 8, 64, mode)
    # real model shapes: time (62 batch x 862 seq), freq (862 x 62)
    attempt('time', 62, 862, 8, 64, 'Half')
    attempt('freq', 862, 62, 8, 64, 'Half')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print('INFER2_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
