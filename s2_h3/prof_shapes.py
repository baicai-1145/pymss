"""Where exactly do the copies and elementwise passes come from: group aten ops by input shapes."""
import os, sys, collections, torch
os.environ.setdefault('S2_DEVICE', 'cpu')
torch.set_num_threads(int(os.environ.get('S2_THREADS', 8)))
sys.path.insert(0, os.environ.get('S2_ROOT', '/home/baicai1145/pymss-bench/.tmp/cpu24/fast'))
import entry
ARM = os.environ.get('ARM', 'H3')
m = entry.build(ARM).eval()
x = torch.randn(1, 2, int(os.environ.get('S2_SAMPLES', 352800)))
with torch.no_grad():
    m(x)
from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        m(x)
tot = 0.0
rows = []
for e in prof.key_averages(group_by_input_shape=True):
    if e.self_cpu_time_total <= 0: continue
    rows.append((e.self_cpu_time_total / 1000.0, e.key, e.input_shapes, e.count))
rows.sort(reverse=True)
print('%-28s %9s %6s  %s' % ('op', 'ms', 'calls', 'input shapes'))
for ms, key, shapes, cnt in rows[:26]:
    print('%-28s %9.1f %6d  %s' % (key[:28], ms, cnt, str(shapes)[:110]))
print('\nWATCHED (copy_/cat/mul/add/div/norm):')
for name in ('aten::copy_', 'aten::cat', 'aten::mul', 'aten::add', 'aten::div', 'aten::linalg_vector_norm', 'aten::index_select', 'aten::contiguous'):
    s = [(ms, shapes, cnt) for ms, k, shapes, cnt in rows if k == name]
    print('  %-26s %8.1f ms   %s' % (name, sum(x[0] for x in s), ' | '.join('%s ×%d' % (str(x[1])[:60], x[2]) for x in s[:3])))
