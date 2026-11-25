# s2_h3 — H3 arm on Ascend 950PR, same protocol as the Colab runs

Snapshot of the Colab experiment directory (`colab_s2`, 2026-11-26) plus device-neutral edits, so that the protocol runs
unchanged on an Ascend NPU (`torch_npu`).  Everything that decides the math is byte-identical to Colab.

| file | role |
|---|---|
| `entry.py` | arms (`ARMS`), model factory, `structure_ok`, `check`; `DEVICE` / `ACCELERATOR` from env `S2_DEVICE` (`npu` = Ascend; unset = CUDA, else CPU), `autocast()`, `install_npu_trainer()` |
| `h3.py` | H3 modules: stacks 0–2 `StaticBandMixer`, stacks 3–5 `SlowBandAttention` + `BlockFeedForward` |
| `stage.py` | one 5k-step segment (train -> metadata asserts -> 56-window eval); flags derived from `DEVICE` |
| `s2_aw_entry.py` | AdamW 5e-4 / StepLR / EMA protocol on top of the pymsst trainer |
| `prepare.py` | MUSDB18-HQ (HF `Viol2000/musdb18hq`, MP3 stems) -> PCM16 WAV tree; root from `S2_ROOT` |
| `run_h2.py` | segment queue: check -> prepare -> `stage.py` per step |
| `protocol_check.py`, `protocol_ref.json` | data identity against the Colab runs (56 window offsets + CRC32 of float16 PCM) |
| `pymsst/`, `pymss_core/` | the package sources Colab imported, verbatim |
| `TASK.md`, `STATUS.md`, `NPU_LOG.md` | work order for the Ascend box, one-line status, every NPU workaround |

What differs from Colab and is *not* protocol: hardware kernels (bf16, SDPA), no g4 fused kernels, no pinned memory.
`protocol_check.py` printing `PROTOCOL_PASS` is the gate before training.

Reference (Colab, 56-window median SI-SDR of the vocals, EMA): H2 5k 5.066 / 10k 6.818 / 15k 7.742 / 20k 7.972 dB.
H3 has to be within 0.3 dB of H2 at 15k to go on to 50k.
