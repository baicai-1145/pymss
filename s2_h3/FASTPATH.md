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

两者合计 3009.9 → 2422.2 ms（同一次会话内测得）。

## 核心放置（同样是 8 个核，1.19×）

这台机器 8 个 NUMA 节点 / 8 个内存控制器，而原协议把 8 个线程全放在节点 0：

| 核心 | `OMP_PROC_BIND` | 时间 / 8 s |
|---|---|---:|
| `0-7`（单节点） | close | 2303.1 ms |
| `0-7`（单节点） | spread | 2284.4 ms |
| `0,8,16,24,32,40,48,56`（8 节点各一核） | close | 1979.7 ms |
| `0,8,16,24,32,40,48,56` | spread | **1941.6 ms** |

收益来自核心集合，不是绑定策略。**但这是协议问题**：单节点才代表消费级 8 核 CPU（一个内存控制器），
分散到 8 节点是"64 核服务器只开 8 线程"。两者差 1.19×，取哪个由部署形态决定，不能默默换成有利的那个。

## 汇总（8 s stereo，权重冻结）

| 配置 | 时间 | RTF | 相对 2769 |
|---|---:|---:|---:|
| 原始基线（8 核单节点） | 2769 ms | 0.346 | 1.00× |
| 代码 + 环境 | 2202 ms | 0.275 | 1.26× |
| 代码 + 环境 + 8 节点分散 | 1941.6 ms | 0.243 | 1.43× |
| 代码 + 环境 + 16 线程 | ~1890 ms | 0.236 | 1.47× |

24 线程不再提升（带宽饱和）。`torch.compile` 实测 1.01×（452 s 编译换 1%），该图 reshape 过多，inductor 不融合。

## 为什么逐元素优化不叠加

单开任一项都是 1.01–1.04×，三项全开仍是 1.04×：瓶颈在节点带宽上转移了。
剖面显示逐元素已是 10–30 GB/s（单 NUMA 节点实测上限），`add` 18 次 137.7 ms、
RMSNorm 的 `pow`+`mul` 118 ms、`copy_` 108.5 ms。**再减少算子数量无用，只能靠融合。**

## 天花板（硬账）

- GEMM：262.6 GFLOP，实测 `addmm` 已到 ~300 GFLOP/s（8 核 Zen2 实用峰值）→ 下限 750–870 ms
- 逐元素/搬运：~900 ms，完美融合后下限 ~400 ms
- 合计下限 ≈ **1.1–1.2 s，即 RTF 0.14**。0.125 需要再砍 20–30% FLOP，或换核数。

## 测量口径（踩过的坑，别再踩）

- 输入长度必须是 352800（8 s = 690 帧）。用训练块 131584 会得到低 3 倍的假数字。
- 候选必须交替跑（ABBA / 单进程 round-robin）。串行跑时后一个吃到预热和频率红利，曾让"channels_last 1.17×"
  这个纯假象通过检查——max|Δ| = 0 恰好说明它什么都没改。
- 用 `torch.no_grad()`，不是 `torch.inference_mode()`：模型里有自定义 autograd 函数会在后者下报错。
- `fastbench.py` 里 `F_ = 1` 的老 bug 让 conv 候选实际跑在 1 行上（标签写着 62），得出假的 1.09 ms。
  真实值 42.1 ms。任何候选都要先确认形状。
- 单位写成"每 8 s 块"，不要写"每 24 s"，混用会差 3 倍。

## 工具

- `verify.py save|cmp ref.npz` —— 同权重同输入，报 max\|Δ\| 与中位耗时
- `ablate.py` —— 单进程交替消融 `FAST` 开关，消掉频率漂移
- `prof_shapes.py` —— 按算子 × 输入形状给出 self 时间
- `fastbench.py conv|model|all` —— 候选实现对比（注意 `BANDS=62`，早前 `F_=1` 的 bug 曾让 conv 候选跑在 1 行上）
