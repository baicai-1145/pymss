# Data gate: measurement of the residual, and the decision it needs

Status: `protocol_check.py` (with `S2_REF_DIR`) → **PROTOCOL_FAIL by the letter of the rule**, on data that the same
script measures as equal to the Colab reference to within **one PCM16 step at 0.07 % of samples**. Training not started.
Everything else is ready (PREFLIGHT_PASS, 50 + 100 tracks decoded, profile done).

## What the per-sample reference says (all 56 windows × 2 stems = 112 comparisons)

| quantity | measured | rule |
|---|---|---|
| samples differing | 0.0727 % mean, **0.1539 % max** | ≤ 2 % → **pass** |
| window energy difference | ≤ 2.4e-6 relative | (no rule) |
| worst single-sample difference, f16 domain | 1 float16 quantum for 66 % of the differing samples, ≤ 8 PCM16 LSB for the rest | ≤ 1 f16 ulp → **fail** |
| worst single-sample difference, absolute | 4.9e-4 = 16 PCM16 LSB, at a sample where the reference is loud (f16 quantum there *is* 16 LSB) | — |
| differing samples whose reference value is exactly 0 | 7 of 57486 | — |

The 48413/57486 = 66 % figure and the rest of the list are in `profile/data_gate_ulp.json` (`protocol_ulp.py`).
Every difference is `±1` in units of the f16 quantum at that sample's magnitude, i.e. the two float16 numbers are
adjacent or one step apart; the exact integer energy check (`E_ref · 2^30` is an integer, so `dS = Σi²_ours − Σi²_ref`
is exact) bounds the *underlying* PCM16 difference to ≈1 LSB on those samples.

## Why the rule as implemented cannot pass

`protocol_check.py` compares `delta ≤ np.spacing(want)`, with `want` the reference rounded to float16. For a quiet
reference sample an f16 step is far *below* one PCM16 step (at `want == 0` it is 1/512 of it), so a difference of one
PCM16 LSB — 3.05e-5, the data's own quantum — is reported as 2…512 "ulp". The criterion is therefore effectively
"bit-identical PCM16" everywhere except in the loudest bins, which no independent decode can satisfy: the reference and
this port quantize the same audio onto two different grids.

Measured in the PCM16 grid (what the wav files hold and what `stage.score()` feeds the model) the two decks agree to
**≤ 1 LSB at ≤ 0.154 % of samples, energy to 2.4e-6**.

## Ruled out as the cause (each: 0/112 CRC matches, no window's energy reproduced)

ffmpeg 4.4.2 (apt, what TASK §2 installs) ≡ ffmpeg 9.0.2 (static) **bit-identically**; ffmpeg `mp3float` vs the
fixed-point `mp3` decoder; ffmpeg CPU/SIMD paths `avx512 / avx2 / avx / sse4.2 / sse4.1 / ssse3 / sse2 / none` (two
distinct outputs, neither matches); ffmpeg's own float→int16 (`-f s16le`, with and without SIMD); rounding rules
even / away / toward-0 / toward-+inf / toward-−inf / truncate at scale 32768 and 32767, in the float32 and float64
domains; libsndfile 1.0.31 (Ubuntu system lib) vs 1.2.2 (pip wheel — the reference is clearly the 1.2.2 family, since
1.0.31 differs by 1 LSB on ~50 % of samples); libmpg123 (libsndfile's mp3 reader) instead of ffmpeg. Reproduce with
`protocol_diag.py` and `protocol_ulp.py`. As TASK §3 says, the exact last-bit rule is not worth chasing further: the
remaining difference is an arithmetic detail of the Colab build.

## Decision needed (one line either way, then I start)

1. **Accept as protocol-equivalent** — the honest summary for the record: *same tracks, same lengths, same audio to
   ≤1 PCM16 LSB at 0.07 % of samples (energy ≤2.4e-6), every f16-visible difference exactly one quantum*. Then I swap
   the gate to the PCM16-grid formulation (1 LSB, ≤2 % of samples, energy ≤1e-5) and launch §7 immediately.
2. **Re-export from Colab with a bit-exact decoder** — if you want 56/56 literally identical, only the Colab side can
   tell me which build produced them (`ffmpeg -version`, `soundfile.__version__`, `soundfile.__libsndfile_version__`).

Everything is staged for immediate launch: the profile (`PROFILE.md`) says the Triton work that matters is the
depthwise-conv weight gradient (496 ms of the 1169 ms step), and §6 kernel development does not need this gate.
