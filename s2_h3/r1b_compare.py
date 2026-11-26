"""Compare r1b_bands.py runs: output/gradient equivalence + step time + launch count, tag vs base."""
import sys

import numpy as np

base = np.load('.tmp/r1b_base.npz')
for tag in sys.argv[1:] or ['grouped', 'grouped_deep']:
    other = np.load(f'.tmp/r1b_{tag}.npz')
    y0, y1 = base['y'], other['y']
    rel_y = float(np.sqrt(((y1 - y0) ** 2).mean() / max((y0 ** 2).mean(), 1e-30)))
    worst_name, worst = None, 0.0
    shared = [k for k in base.files if k.startswith('g:') and k in other.files]
    for k in shared:
        a, b = base[k], other[k]
        den = max(float(np.sqrt((a ** 2).mean())), 1e-30)
        r = float(np.sqrt(((b - a) ** 2).mean()) / den)
        if r > worst:
            worst_name, worst = k[2:], r
    missing = [k[2:] for k in base.files if k.startswith('g:') and k not in other.files]
    m0, m1 = base['meta'], other['meta']
    print(f'--- {tag} vs base ---')
    print(f'  fwd rel_rms      {rel_y:.3e}')
    print(f'  worst grad rel   {worst:.3e}  at {worst_name}')
    print(f'  grads missing    {len(missing)} {missing[:3]}')
    print(f'  loss             {m0[2]:.8f} -> {m1[2]:.8f}')
    print(f'  step ms          {m0[0]:.1f} -> {m1[0]:.1f}  ({m0[0]/m1[0]:.2f}x)')
    print(f'  launches         {int(m0[1])} -> {int(m1[1])}  ({(1-m1[1]/m0[1])*100:+.1f}%)')
    print(f'  hbm GB           {m0[3]:.1f} -> {m1[3]:.1f}')
