"""Attribute every forced copy to the user-code line that caused it."""
import collections, os, sys, traceback, torch
os.environ.setdefault('S2_DEVICE', 'cpu')
torch.set_num_threads(int(os.environ.get('S2_THREADS', 8)))
sys.path.insert(0, os.environ.get('S2_ROOT', '/home/baicai1145/pymss-bench/.tmp/cpu24/fast'))
ROOT = os.environ.get('S2_ROOT', '')
rec = collections.Counter()
orig = torch.Tensor.contiguous
def patched(self, *a, **k):
    out = orig(self, *a, **k)
    if self.numel() >= 1 << 20 and out.data_ptr() != self.data_ptr():
        site = '?'
        for fr in reversed(traceback.extract_stack()[:-1]):
            if ROOT in fr.filename and 'copy_audit' not in fr.filename:
                site = '%s:%d' % (os.path.basename(fr.filename), fr.lineno)
                break
        rec[(site, tuple(self.shape), tuple(self.stride()))] += self.numel() * 4
    return out
torch.Tensor.contiguous = patched
import entry
m = entry.build(os.environ.get('ARM', 'H3')).eval()
x = torch.randn(1, 2, int(os.environ.get('S2_SAMPLES', 352800)))
with torch.no_grad(): m(x)
rec.clear()
with torch.no_grad(): m(x)
torch.Tensor.contiguous = orig
tot = sum(rec.values())
print('forced contiguous copies: %.0f MB total' % (tot / 1e6))
for (site, shp, st), b in rec.most_common(10):
    print('  %-24s %-20s %6.0f MB   stride %s' % (site, str(shp), b / 1e6, st))
