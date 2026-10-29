"""Isolated per-shape launch test for supa_rope_apply (failed launches poison the SUPA context).

Each shape runs in its own subprocess so one invalid-parameter failure cannot
corrupt the rest of the sweep. Primary hypothesis: batch*seq (rows) capped near
65536. Verifies the boundary and the chunked shapes that would let the model
use the op (time side (62,862), freq side (862,62), both 53,444 rows).
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CHILD = r'''
import json, sys, torch
batch, seq, dim = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
from pymss_core.device import prepare_device_alias
prepare_device_alias('biren')
device = torch.device('supa:0')
try:
    x = torch.randn(batch, seq, dim, device=device, dtype=torch.bfloat16)
    half = torch.randn(1, seq, 1, dim // 2, device=device)
    cf = torch.cat([half, half], dim=-1).reshape(1, seq, dim).contiguous()
    out = torch.ops.supa.supa_rope_apply(x, cf, cf.clone())
    torch.supa.synchronize()
    print(json.dumps({'ok': True, 'finite': bool(out.isfinite().all()),
                      'layout': str(getattr(out, 'br_layout', None))}))
except Exception as error:
    print(json.dumps({'ok': False, 'error': str(error).splitlines()[-1][:140]}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.tmp/br-rope-sweep2.json')
    args = parser.parse_args()
    shapes = [
        (3855, 17, 64),   # 65,535 rows: pass iff cap is exactly 65,536
        (3856, 17, 64),   # 65,552 rows
        (2048, 17, 64),
        (496, 862, 64),   # model time shape (427,152 rows) - reproduce model failure
        (62, 862, 64),    # time chunk candidate (53,444 rows)
        (124, 862, 64),   # 106,888 rows
        (6896, 62, 64),   # model freq shape (427,552 rows)
        (862, 62, 64),    # freq chunk candidate (53,444 rows)
        (1724, 62, 64),   # 106,888 rows
        (16, 4096, 64),   # 65,536 rows exactly
    ]
    results = []
    for batch, seq, dim in shapes:
        try:
            run = subprocess.run([sys.executable, '-c', CHILD, str(batch), str(seq), str(dim)],
                                 capture_output=True, text=True, timeout=90,
                                 env={**os.environ, 'PYMSS_SDPA_BACKEND': 'sudnn'})
            line = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else '{}'
            record = json.loads(line)
        except subprocess.TimeoutExpired:
            record = {'ok': False, 'error': 'child timeout'}
        record.update({'batch': batch, 'seq': seq, 'dim': dim, 'rows': batch * seq})
        results.append(record)
        print(f"SWEEP2 {'PASS' if record['ok'] else 'FAIL'} b={batch} s={seq} rows={batch*seq} "
              f"{record.get('layout', record.get('error', ''))}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print('SWEEP2_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
