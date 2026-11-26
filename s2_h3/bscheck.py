import os, sys
os.environ['S2_DEVICE'] = 'cpu'; os.environ.setdefault('S2_ROOT', '/workspace/s2_h3/s2_h3')
sys.path.insert(0, '/workspace/s2_h3/s2_h3')
import torch
from entry import build
groups = {}
for arm in ('A', 'H2', 'H3'):
    m = build(arm).eval()
    named = dict(m.named_parameters())
    def total(pred): return sum(p.numel() for k, p in named.items() if pred(k))
    bs = total(lambda k: k.startswith('band_split'))
    fn = total(lambda k: k.startswith('final_norm'))
    mh = total(lambda k: k.startswith('to_freq') or k.startswith('mask') or '_estimate_masks' in k or k.startswith('to_audio'))
    other = sum(p.numel() for p in m.parameters()) - bs - fn - mh
    tot = sum(p.numel() for p in m.parameters())
    groups[arm] = (tot, bs, fn, mh, other)
    print('%-3s total %9d | band_split %9d | final_norm %6d | mask_head %9d | rest %9d' % (arm, tot, bs, fn, mh, other))
same = groups['A'][1:4] == groups['H2'][1:4] == groups['H3'][1:4]
print('BS_CHAIN_IDENTICAL', same, '| band_split', groups['A'][1], '| final_norm', groups['A'][2], '| mask head', groups['A'][3])
# byte-level: same shapes and same initial values? all arms re-seed with torch.manual_seed(0) in build()
vals = {}
for arm in ('A', 'H2', 'H3'):
    m = build(arm)
    d = {k: v.detach().clone() for k, v in m.named_parameters() if k.startswith(('band_split', 'final_norm'))}
    vals[arm] = d
ks = sorted(vals['A'])
diff = max(float((vals['A'][k] - vals[arm][k]).abs().max()) for arm in ('H2', 'H3') for k in ks)
print('band_split+final_norm: %d tensors, max |A - {H2,H3}| = %.1e' % (len(ks), diff))
