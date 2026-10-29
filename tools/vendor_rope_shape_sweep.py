"""Find the shape contract of supa_rope_apply by launch sweep.

Validated at (16,17,64) but model shapes (496,862,64)/(6896,62,64) fail with
SUPA Kernel Launch Error invalid parameter. Isolate which axis constrains the
launch: sweep batch, seq and dim independently around the model shapes.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.tmp/br-rope-sweep.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    device = torch.device('supa:0')
    rope = torch.ops.supa.supa_rope_apply
    results = []

    def attempt(batch, seq, dim):
        x = torch.randn(batch, seq, dim, device=device, dtype=torch.bfloat16)
        half = torch.randn(1, seq, 1, dim // 2, device=device)
        cf = torch.cat([half, half], dim=-1).reshape(1, seq, dim).contiguous()
        sf = cf.clone()
        try:
            out = rope(x, cf, sf)
            torch.supa.synchronize()
            finite = bool(out.isfinite().all())
            results.append({'batch': batch, 'seq': seq, 'dim': dim, 'ok': True, 'finite': finite,
                            'out_shape': list(out.shape), 'out_layout': str(getattr(out, 'br_layout', None))})
            print(f'SWEEP_PASS b={batch} s={seq} d={dim} finite={finite} layout={getattr(out, "br_layout", None)}', flush=True)
        except RuntimeError as error:
            message = str(error)
            results.append({'batch': batch, 'seq': seq, 'dim': dim, 'ok': False,
                            'error': message.splitlines()[-1][:160]})
            print(f'SWEEP_FAIL b={batch} s={seq} d={dim} :: {message.splitlines()[-1][:120]}', flush=True)
        finally:
            del x, cf, sf

    # isolate seq with tiny batch, isolate batch with tiny seq
    for seq in (17, 62, 63, 64, 128, 256, 512, 600, 768, 832, 860, 862, 864, 1024, 2048):
        attempt(16, seq, 64)
    for batch in (16, 62, 128, 256, 496, 512, 1024, 4096, 6896, 8192):
        attempt(batch, 17, 64)
    # model shapes and near neighbors
    attempt(496, 862, 64)
    attempt(6896, 62, 64)
    for dim in (32, 64, 128):
        attempt(16, 17, dim)
    # padding hypotheses: seq rounded up, batch rounded down
    attempt(496, 864, 64)
    attempt(495, 862, 64)
    attempt(384, 862, 64)
    attempt(256, 862, 64)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print('SWEEP_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
