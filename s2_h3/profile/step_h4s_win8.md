# Win 8 increment 1 — the 58 ms is real work, not bubbles

Async timeline over **20 clean steps** with nothing instrumented inside the step (one `torch.profiler` NPU trace,
`cat=kernel` events only, 92,200 kernels, all on a single stream so `sum(dur)` is exact busy time).

## Result

| quantity | value |
|---|---|
| kernel-busy per step | **216.2 ms** |
| timeline span per step | 216.7 ms |
| **busy fraction** | **99.8 %** |
| total gap over 20 steps | **8.9 ms** = **0.45 ms/step** |
| kernels per step (device) | 4,610 |
| largest single gap | 0.16 ms |

**Deciding number: busy/span = 99.8 %.** There are no device bubbles to schedule away; the largest gap in twenty steps is
0.16 ms. So the ~58 ms that `sum(self_device_ms)` left unattributed is **real device work that the per-op ranking
undercounts**, and the ranking approach stands.

Cross-check that closes the budget: 216.2 ms kernel + ~6 ms host = **222.1 ms wall**, which is the measured step. The
`self_device_ms` sum (163.5 ms) is therefore a *ranking of a subset*, not a budget — it drops ~52 ms, most likely
sub-threshold ops and the per-op self-time accounting.

## The correct ranking (complete, from the full trace)

| family | ms/step | calls/step |
|---|---|---|
| **Add** | **31.1** | 73 |
| **MatMulV** | **21.8** | 708 |
| **Transpose** | **19.4** | 250 |
| **Slice** | **15.2** | 405 |
| **Cast** | **15.0** | 1,303 |
| DropOutV | 11.9 | 56 |
| RmsNormGrad | 10.6 | 81 |
| `_dw_ntc_tiled` | 10.0 | 4 |
| Conv | 9.6 | 8 |
| ReduceSum | 9.1 | 268 |
| BatchMatMulV | 8.0 | 69 |
| ConcatD | 6.6 | 20 |

Top individual kernels: `Slice` 15.2, `Cast` 15.0, `Add_FLOAT32_8` 10.9, `_dw_ntc_tiled` 10.0, `RmsNormGrad` 9.0,
`Conv2DV2` 8.8, `Add_FLOAT32_9` 8.3, `DropOutV3` 8.1, `MatMulV3_bf16` 8.0, `Add_BFLOAT16_FLOAT32_8` 7.7.

Two corrections to the earlier picture: **Add is the single largest family at 31.1 ms over only 73 calls** (0.43 ms per
call — these are full-tensor passes, not launch-bound), and **Transpose at 19.4 ms / 250 calls** did not appear at all in
the `self_device_ms` top-10, so layout churn is a bigger lever than that ranking suggested.

**Conclusion: device-bound at 216 ms with 99.8 % occupancy. Kernel fusion / layout elimination is the path; scheduling is
not.** Next increments per the brief: `_dw_ntc_tiled` transposes (10.0 ms kernel, 2 transposes per conv), then the Cast
family after answering the autocast question.

## Increment 3 — the Cast question, answered before writing any kernel

Cast is **15.0 ms over 1,303 calls (0.0115 ms/call)** — launch-cheap. Producer attribution:

| cast block | ms | pattern |
|---|---|---|
| after `Add_FLOAT32` -> before `ZerosLike_2_BYTES` | 4.2 | structural, not dtype churn |
| `Cast -> Cast` chains | 3.2 / 2.2 | structural |
| before bf16 `MatMulV3` | **1.9** | genuine autocast churn |
| before `DropOutV3Grad` | 1.0 | dropout mask dtype |

**So dtype pinning (Win 6) would remove at most ~2-6 ms of the 15.0**, and the rest is structural — not worth a
checkpoint-dialect change, and not worth a fused Cast kernel at 0.0115 ms/call. Cast is closed as a target.

## Increment 3b — where the leverage actually is: layout chains feeding Add

Attributing the biggest family by its neighbours instead of by name:

| family | ms/step | calls | ms/call | dominant chain |
|---|---|---|---|---|
| **Add** | **32.5** | 628 | 0.052 | after `ViewCopy` 9.3, after `GatherV2` **8.4**, after `BroadcastTo` 6.9 |
| Transpose | 19.4 | 250 | 0.078 | runs into `Slice` (3.2) and `RotaryPositionEmbeddingGrad` |
| Slice | 15.2 | 405 | 0.038 | **after `Transpose` 12.0 -> before `ViewCopy` 12.0** |

One chain dominates: **`Transpose -> Slice -> ViewCopy -> Add`, plus `GatherV2 -> Add` (8.4) and
`BroadcastTo -> Add` (6.9)** — roughly **37 ms/step** of copy-then-add that a single fused
"gather/broadcast + add" kernel collapses into one pass, with no intermediate tensor. That is the next target, and it is
the fused `block_broadcast_add` the brief anticipated, now with a measured address instead of a guess.
