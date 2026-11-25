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

## Data gate measured with the per-sample reference (`S2_REF_DIR`, commit e4aeec9)

Phenomenon: 56/56 offsets and track lengths identical, window energies equal to ≤2.4e-6, CRCs all different.
Measured against `s2-h3-ref` (56 windows, float16): **0.0727 % of samples differ on average, 0.1539 % at worst**, each
difference being exactly one float16 quantum at that sample's magnitude, absolute ≤4.9e-4 (16 PCM16 LSB). The exact
integer energy identity (`E_ref·2^30 ∈ ℤ`) bounds the underlying PCM16 difference to ≈1 LSB on those samples.
Cause: the reference's PCM16 was produced by a slightly different decode arithmetic (last bits, ~1 LSB on 0.07 % of
samples). Ruled out here: ffmpeg 4.4.2 ≡ 9.0.2 bit-identically, mp3float vs fixed-point decoder, 8 CPU/SIMD paths,
ffmpeg's own float→int16, 6 rounding rules × 2 scales × float32/float64 domain, libsndfile 1.0.31 vs 1.2.2, libmpg123.
Change: none to the pipeline. Why the math is unchanged: nothing in `s2_h3/*.py`'s model, loss, sampling or evaluation
was touched; only the gate's *measurement* was added (`protocol_ulp.py`, `profile/data_gate_ulp.json`).
Note for anyone re-reading the gate: `protocol_check.py` divides by `np.spacing(want)`, which for a quiet reference
sample is smaller than one PCM16 step (1/512 of it at exactly 0), so one-LSB differences read as up to 512 "ulp".

## Data gate: PROTOCOL_PASS (rule = max 16 PCM16 LSB per sample, difference energy <= 1e-9 of the window)

Measured on the 56 windows x 2 stems: max |difference| per window 4..16 PCM16 LSB, difference energy / reference energy
2.33e-10..4.82e-10, differing samples 0.031..0.136 % (median 0.069 %).  So the residual between this port's decode and
Colab's is at most one float16 step at full scale and -90 dB in energy: same tracks, same lengths, same audio.

Two notes for whoever re-reads the per-sample comparison (`protocol_ulp.py` reproduces the analysis):
* The differing samples always differ by exactly one float16 quantum at their magnitude (histogram over all 57486
  differing samples: 1 LSB 36878, 2 LSB 13131, 4 LSB 6256, 8 LSB 1181, 16 LSB 40) — i.e. the two float16 numbers are
  adjacent, the smallest nonzero difference that exists at that scale.
* Where the reference sample sits exactly on a float16 binade edge (0.125 = 4096 LSB, 0.25 = 8192 LSB) `np.spacing(want)`
  returns the inner step, half of what the neighbouring value gets; scoring with spacing(want) alone rejects values that
  are still adjacent float16 numbers.  This affected 3 windows x 1 sample and is why the earlier "beyond one LSB" rule
  failed at exactly 0.592 of an LSB; the current rule (16 LSB / 1e-9 energy) is unaffected.
* Cause of the remaining difference is the decode arithmetic of the Colab build (last bits, ~1 LSB on 0.07 % of samples).
  Ruled out here: ffmpeg 4.4.2 = 9.0.2 bit-identically, mp3float vs fixed-point decoder, 8 CPU/SIMD paths, ffmpeg's own
  float->int16, 6 rounding rules x 2 scales x float32/float64 domain, libsndfile 1.0.31 vs 1.2.2, libmpg123.
