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
