"""Pinpoint the remaining copies/cats by python call stack."""
import os, sys, torch
os.environ.setdefault('S2_DEVICE', 'cpu')
torch.set_num_threads(int(os.environ.get('S2_THREADS', 8)))
sys.path.insert(0, os.environ.get('S2_ROOT', '/home/baicai1145/pymss-bench/.tmp/cpu24/fast'))
import entry
m = entry.build(os.environ.get('ARM', 'H3')).eval()
x = torch.randn(1, 2, int(os.environ.get('S2_SAMPLES', 352800)))
with torch.no_grad(): m(x)
from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], with_stack=True, record_shapes=True) as prof:
        m(x)
agg = {}
for e in prof.key_averages(group_by_stack_n=6):
    if e.key not in ('aten::copy_', 'aten::cat', 'aten::add', 'aten::mul') or e.self_cpu_time_total < 2000: continue
    st = ' <- '.join(f'{f.name}:{f.lineno}' for f in (e.stack or [])[:5] if 'torch/' not in (f.filename or ''))
    agg[(e.key, e.input_shapes, st)] = (e.self_cpu_time_total / 1000.0, e.count)
for (k, shp, st), (ms, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:12]:
    print('%-11s %7.1f ms x%-3d %s\n            %s' % (k, ms, n, str(shp)[:60], st[:150]))
