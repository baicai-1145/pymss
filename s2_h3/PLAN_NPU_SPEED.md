# H4s speed sprint: stage 5000 → 10000 (target ~10 steps/s, i.e. ~100 ms/step)

Baseline: **371.1 ms/step (2.70 steps/s)**, HBM 17.3 GB, NPU util 76 % — measured with `S2_TRITON=dwconv_dw_bmm`
before training started. Reference points: H3 eager 1112.6, H3 shipped 604.9. Target is a **3.7× further** reduction,
so no single item can carry it; the plan is ordered by (expected gain ÷ risk), each item gated identically.

**Guardrails.** The running stage 5000 is not touched. `run_h2.py` is SIGSTOPped (`Ts`) so stage 10000 cannot
auto-start; `stage.py`/`s2_aw_entry.py` keep running in their own session and finish 5000 + eval + export normally.
All benchmarks run in the exclusive window **after** `STAGE_OK H4s 5000`, and the queue is relaunched only with the
admitted configuration. Nothing on the NPU while a stage is training.

## Verified before planning (static, no device use)

| claim | status |
|---|---|
| `ASCEND` profile inherits `fft_result_check=True` | **confirmed** — `pymss_core/device.py:244` (`TPU` sets it `False` explicitly at `:285`) |
| `_degenerate()` does a host sync per STFT/ISTFT | **confirmed** — `npu_fft.py:159` `float(out_planes.abs().max())`, called at `:449` (rfft) and `:504` (irfft) |
| native fused ops exist in torch_npu 2.13.0 | **confirmed** — `npu_swiglu`, `npu_rotary_mul`, `npu_rms_norm`, `npu_add_rms_norm`, `npu_fast_gelu`, `npu_gelu_mul`, `npu_apply_rotary_pos_emb`, `npu_fusion_attention` (in `utils/exposed_api.py` / `_op_plugin_docs.py`), `torch_npu.optim.NpuFusedAdamW` |
| Triton-Ascend UB-slice API exists | **confirmed** — `language/extra/cann/extension/{vec_ops,core}.py` expose `extract_slice`, `insert_slice`, `conv1d` (with `groups=`, `padding_size=`) |
| `tl.load(..., care_padding=False)` | **absent in this wheel** — no `care_padding` anywhere under `triton/language/`; will not be used |

## Round plan

### R1. Kill the per-step D2H sync (highest value, lowest risk)
Change: cache vendor-FFT trust per `n_fft` in `npu_fft.py` (`_VENDOR_FFT_OK` set) so `_degenerate` runs **at most once
per shape**, and set `fft_result_check=False` for `ASCEND` in `device.py` (the TPU entry documents exactly this
reasoning: "two full reductions and two host syncs per STFT and a `torch.compile` graph break").
Why first: it is 2 syncs/step *and* the precondition for R4 (no graph can be captured across a `.item()`).
Gate: numerics unchanged (it removes a check, not math) + the existing `entry.py check` + step timing. Also verify once
offline that `torch.fft.rfft`/`irfft` on this device are not degenerate at the model's `n_fft`, so the cache is honest.

### R2. Graph capture (largest expected win; needs R1)
H4s step has ~1000+ small ops at only 76 % NPU util → host dispatch bound. Test in order, on the **model core**
then the whole step: `torch.compile(..., backend='npugraphs')`, `backend='inductor', mode='reduce-overhead'`,
`backend='npugraph_ex'`. Lightning wraps the model, so compile `pl_module.model` (the mask-core callable), not the
LightningModule. Gate: 200-step loss curve vs the H4s eager reference ≤ 1 % drift, plus identical 56-window eval
protocol; timing on the same step.
Risk: ACLGraph needs static shapes (true: batch 24 × 131584) and no host syncs (R1 removes the last ones); if capture
fails, record the error and fall back to R3/R5 rather than fighting it.

### R3. Native fused ops (moderate gain, low risk, exact autograd)
1. **RoPE** — `torch_npu.npu_rotary_mul(x, cos_full, sin_full, rotary_mode='interleave')` replaces
   `apply_rotary_emb_fast` (4 strided muls + 2 strided copies; 61.6 ms/step in the H3 profile). Need
   `cos_full/sin_full = repeat_interleave(2, -1)` and a shape/layout check (BSND vs BNSD) → build a small equivalence
   test against the current function *including gradients* before wiring it in.
2. **SwiGLU** — `torch_npu.npu_swiglu` computes `silu(first) * second`, MSC computes `first * silu(second)`.
   **Resuming from the 5k checkpoint forbids changing semantics**, so the halves of `proj_in.weight`/`bias` are swapped
   **once at load** (row permutation of the output channels), which is an exact re-parameterisation, verified by
   comparing one forward against the un-permuted model to 0 rel error. If the permutation cannot be made exactly
   (checkpoint sharding), keep `u * F.silu(v)` and skip this op.
3. **RMSNorm** — already `F.rms_norm`; measure `npu_rms_norm` vs it, and try `npu_add_rms_norm` where a residual add
   precedes the norm (removes one `Add`).
4. **Optimizer / EMA** — `torch_npu.optim.NpuFusedAdamW` replaces ~500 per-param launches; EMA
   (`AveragedModel.update_parameters`) currently a Python per-param loop → `torch._foreach_lerp_`/`_foreach_mul_`.
   Gate: optimizer state must be bit-identical to the current one on a 20-step run (same seed, same data order).

### R4. Triton, done the Ascend way (for the conv path that still costs ~118 ms in H3's profile)
Persistent grid `(56,)` with an inner loop, and the **load-once-into-UB** layout instead of the 3-D `(BC,K,BT)` tile:
one padded row is `T+K-1 = 288` elements = 576 B; a `(16, 288)` bf16 tile is 9.2 KB, so double-buffering is free.
Inside UB, walk the taps with `extension.extract_slice(buf, (0, k), (BC, T), (1, 1))` — no negative-offset scattered
DMA, no im2col, no `AsStrided`, no 31× re-read. Same pattern gives dX and dW (dW accumulates `(BC,K)` in fp32 per core
and writes once, or into a `[56, C, K]` buffer summed by a second tiny kernel — no atomics).
Immediate cheap baseline first: `S2_TRITON_BMM_CHUNK` 64/128/256 on the current bmm dW (H4s has only 17.3 GB in use).
Gate: the §6 admission test (fp32 vs float64 ≤ 1e-5, bf16 vs eager ≤ 1e-2, faster than the incumbent) + step timing.

### R5. Remaining hotspots from the fresh H4s profile
Re-profile after each admitted round (`profile/step_h4s_roundN.json`), handle items ≥ 25 ms largest-first, same gates.

## Measurement protocol (each round, exclusive NPU)

1. `profile_step_npu.py` with `ARM=H4s S2_TRITON=<admitted set> S2_TAG=h4s_roundN` → whole step + operator table.
2. Step rate: 10 warmup + 20 timed steps, median (`steprate` harness), HBM peak, NPU util.
3. Correctness: for kernels/ops an equivalence test (fp32/fp64 as applicable); for anything affecting the training
   trajectory, a **20-step bit-comparison** and then a **200-step loss-curve control ≤ 1 %** against
   `runs/H4s-ref-eager` (to be produced once, exactly like `runs/H3-ref-eager`).
4. Record: `profile/step_h4s_roundN.json`, a row in `TRITON.md`, and the final kernel list in the launch command.

## Reporting

Each round: whole-step ms, ×vs eager (1112.6) and ×vs the 371 ms H4s baseline, the new profile table, and what was
rejected and why. Stop at ~100 ms/step, or when no item ≥ 25 ms remains, or after 3 consecutive failed admissions.
