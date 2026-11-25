# PROTOCOL_FAIL on the data gate — what differs, how big it is, what to do

`protocol_check.py` → `PROTOCOL_FAIL` on 2026-11-27 on the Ascend box.  Training was **not** started (TASK §3).
Everything else in the pipeline is ready: `PREFLIGHT_PASS` on the NPU, 50 test + 100 train tracks decoded (110 GB).

## What matches

* 56/56 reference windows are reached: `random.Random(0xC0FFEE + epoch*100003 + slot)` lands on the same
  `(name, off)` in both runs, so the track list and every track length are identical → the MP3 archive is the
  one the Colab runs used (`Viol2000/musdb18hq`, same `test.tar`).
* `energy_*` in `protocol_ref.json` is an **exact integer** multiple of 2^-30, i.e. it is literally
  `sum((i/32768)**2)` over the reference's int16 samples.  So the reference is PCM16 of *this* audio, not of
  another dataset revision.

## How big the difference is

| quantity | value |
|---|---|
| relative window-energy difference, ours vs reference | 7.5e-9 … 2.4e-6 (median ~1e-6) |
| implied per-sample difference RMS | ≲ 1e-3 of one PCM16 LSB (a few float32 ULP), i.e. ≈ -120 dB |
| PCM16 quantization floor of the data itself | 1 LSB = 3.05e-5 (≈ -90 dB) |

Both stems of every window are affected equally, in both signs, so this is not a scale, offset or resampling
error: our decoded float32 samples differ from Colab's in the last few bits, which moves a small percentage of
samples to the other side of a rounding boundary.  A 352800×2 window CRC is ~1e6 bits wide, so any single such
sample breaks it — hence 0/56 bit-identical even though the audio is the same to -120 dB.

## Ruled out (each 0/12 CRCs and never the exact reference energy), see `protocol_diag.py`

| candidate | result |
|---|---|
| ffmpeg 4.4.2 (apt, as TASK §2 asks) vs ffmpeg 9.0.2 (static) | **bit-identical output** → not a version drift |
| ffmpeg `mp3float` vs fixed-point `mp3` decoder | different, neither matches |
| ffmpeg CPU paths `avx512 / avx2 / avx / sse4.2 / sse4.1 / ssse3 / sse2 / none` | 2 distinct outputs, neither matches |
| ffmpeg writing s16 itself (`-f s16le`, SIMD and non-SIMD) | different, neither matches |
| float→int16 rules: even / away-from-zero / toward-zero / toward +inf / toward -inf / truncate, scale ×32768 and ×32767, float32 and float64 domain | none matches |
| libsndfile 1.0.31 (Ubuntu system lib) vs 1.2.2 (pip wheel, what we run) | they differ by 1 LSB on **~50 %** of samples → too large to be the reference; the reference is much closer to 1.2.2 |
| libmpg123 (libsndfile 1.2.x mp3 reader) instead of ffmpeg | right order of magnitude (RMS 5e-8 vs ffmpeg) but not the reference |

So the reference's arithmetic is a combination none of the toolchains available on this box reproduces.  The
remaining unknowns are all on the Colab side of the wire.

## What I need from the lead (cheapest first)

1. **Colab's toolchain fingerprint**: `ffmpeg -version | head -3`, `python -c "import soundfile; print(soundfile.__version__, soundfile.__libsndfile_version__)"`,
   `numpy.__version__` — from the session that produced the 2026-11-24/25 refs.
2. **A sample diff**, if the fingerprint alone doesn't settle it: either one track's decoded `vocals.wav`
   (~38 MB, or just its first 200 kB) or, much cheaper, the first 64 int16 samples of
   `AM Contra - Heart Peripheral.vocals.wav` at offset 2908455 (window 0 of the ref).  I print ours for the same
   position; a one-off `np.fromfile(...)` on the Colab side is enough to see whether the samples are equal,
   off by one, or shifted.
3. **Or accept the residual as protocol-neutral.**  -120 dB is ~30 dB below the data's own PCM16 quantization
   floor; the 56-window median SI-SDR cannot move in the 0.01 dB digit from it, so H3-vs-H2 at 15k stays
   comparable.  In that case say so and I train immediately (`PROTOCOL_PASS` would be replaced by a documented
   `PROTOCOL_ACCEPTED_ULP` note, not silently skipped).

Ours, for the record — int16, first 12 samples of `AM Contra - Heart Peripheral.vocals.wav` at 2908455:
`[[1,0],[0,0],[0,0],[-1,0],[0,-1],[0,0],[0,0],[0,-1],[0,0],[0,0],[0,0],[0,0]]`
