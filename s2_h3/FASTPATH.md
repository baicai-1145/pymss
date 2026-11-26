# H3 CPU 推理快路径：实测记录

协议：NAS 上的 EPYC 7532，`taskset -c 0-7`（单 NUMA 节点），torch 2.15.0+cpu / py3.11.2，FP32，batch 1，
输入 1×2×352800（8 s stereo，690 帧），权重冻结。基线 2769 ms（= 记录值 2.7700 s），RTF 0.346。

## 有效的改动（全部数学等价，逐项验证过）

| 改动 | 位置 | 效果 | 数值 |
|---|---|---|---|
| 深度卷积改走 2-D channels_last | `entry.py::conv_forward` | 42.1 → 11.6 ms/次（4 次调用） | max\|Δ\| = 0 |
| 丢掉 conv 前多余的 `contiguous()` | `entry.py::MSC.forward` | −110 ms | — |
| MSC 每层少一次跨步拷贝，`proj_out` 走 matmul | `entry.py::MSC.forward` | 每层 165 → 99 MB 流量 | 梯度 rel 8.4e-6 |
| `SlowBandAttention` 去掉 267 MB 广播 | `h3.py` | −42 ms，3 层各一次 | max\|Δ\| = 0 |
| 注意力输出保持连续布局 | `pooled_kv.py`, `entry.py` | −83 ms | rel 5.4e-07 |
| RMSNorm 用融合核 | `transformer.py::RMSNorm` | −110 ms | rel 4.9e-07 |
| RoPE 去 `cat` | `transformer.py::apply_rotary_emb_fast` | −80 ms | max\|Δ\| = 0 |

代码改动合计：2769 → 2202 ms。三个运行期开关在 `transformer.FAST` 里，`ablate.py` 可在单进程内交替消融。

## 零代码的环境收益（更大）

| 设置 | 效果 |
|---|---|
| `OMP_PROC_BIND=close OMP_PLACES=cores` | 线程原来跨 8 个 NUMA 节点漂移；−1.24× |
| `MALLOC_MMAP_THRESHOLD_ / MALLOC_TRIM_THRESHOLD_` 调大 | 大张量不再每次 mmap+缺页，改复用堆；−1.08× |

两者合计 3009.9 → 2422.2 ms（同一次会话内测得）。最终 8 核 **2202 ms，RTF 0.275**。
16 线程/2 节点 → 1890 ms（RTF 0.236）；24 线程不再提升。

## 为什么逐元素优化不叠加

单开任一项都是 1.01–1.04×，三项全开仍是 1.04×：瓶颈在节点带宽上转移了。
剖面显示逐元素已是 10–30 GB/s（单 NUMA 节点实测上限），`add` 18 次 137.7 ms、
RMSNorm 的 `pow`+`mul` 118 ms、`copy_` 108.5 ms。**再减少算子数量无用，只能靠融合。**

## 天花板（硬账）

- GEMM：262.6 GFLOP，实测 `addmm` 已到 ~300 GFLOP/s（8 核 Zen2 实用峰值）→ 下限 750–870 ms
- 逐元素/搬运：~900 ms，完美融合后下限 ~400 ms
- 合计下限 ≈ **1.1–1.2 s，即 RTF 0.14**。0.125 需要再砍 20–30% FLOP，或换核数。

## 工具

- `verify.py save|cmp ref.npz` —— 同权重同输入，报 max\|Δ\| 与中位耗时
- `ablate.py` —— 单进程交替消融 `FAST` 开关，消掉频率漂移
- `prof_shapes.py` —— 按算子 × 输入形状给出 self 时间
- `fastbench.py conv|model|all` —— 候选实现对比（注意 `BANDS=62`，早前 `F_=1` 的 bug 曾让 conv 候选跑在 1 行上）
