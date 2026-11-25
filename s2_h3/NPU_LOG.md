# NPU log

Every workaround needed to run the Colab protocol on the Ascend 950PR: symptom, cause, change, why the math is unchanged.

## Data gate: PROTOCOL_FAIL is a decode-arithmetic difference, not an NPU issue

Symptom: `protocol_check.py` → 0/56 windows bit-identical on the NPU box, while all 56 window offsets match
(same tracks, same lengths) and each window's energy matches to 7.5e-9 … 2.4e-6 relative.
Cause: the reference's PCM16 is the same recording decoded with slightly different float arithmetic
(≲1e-3 of a PCM16 LSB per sample, ≈ -120 dB); a 352800×2 CRC cannot tolerate that.
Change: none — training not started (TASK §3). `protocol_diag.py` + `PROTOCOL_FAIL_REPORT.md` record what was
ruled out (ffmpeg 4.4.2 ≡ 9.0.2 bit-identically, mp3float vs fixed-point, 8 CPU/SIMD paths, ffmpeg's own
float→int16, 6 rounding rules × 2 scales, libsndfile 1.0.31 vs 1.2.2, libmpg123).
Why the math is unchanged: nothing in `s2_h3/*.py`'s model/optimizer/data pipeline was touched; only a
diagnostic script was added.

## Snapshot gap: `pymsst/data/` was missing, so the training entry could not start

Symptom: `python -c "import pymsst.cli"` → `ModuleNotFoundError: No module named 'pymsst.data'`; `s2_aw_entry.py` runs
`runpy.run_module('pymsst')`, which imports `cli.py` → `.data.musdb`.  The snapshot shipped `pymsst/{cli,export,training,
validation}` but not the `data` package.
Cause: files omitted from the snapshot; the pinned package at `/workspace/pymsst@6d01513` has the same `cli.py`
(622 lines, differing only in the `g4` fused-backend option) and the matching `data/`.
Change: copied `pymsst/data/{__init__,audio,datamodule,musdb}.py` **verbatim** from `/workspace/pymsst/src/pymsst/data`
(read-only copy, that worktree untouched), force-added because the repo `.gitignore` has a bare `data` pattern.
Why the math is unchanged: same files the Colab package version imports; no chunking, sampling or augmentation code was
edited.

## Profiling notes (TASK §5)

* Two identical `model(mixture)` calls differ by 1.7e-07 relative on this device, so "identical computation" checks have
  to use a tolerance (1e-6 here) instead of bitwise equality — also applies to the §6 numeric admission tests.
* `torch.profiler` needs `activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]`; per-op device time is
  `self_device_time_total`.
* The whole H3 step is 1169 ms: forward 150, backward 959, optimizer+EMA 18, h2d 59.  `Conv3DBackpropFilterV2` (the
  depthwise-conv **weight** gradient, 12 calls) is 496 ms of it; on one conv at the training shape the split is
  forward 3.7 / dX 8.2 / **dW 122.9 ms**.
