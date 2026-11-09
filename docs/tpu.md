# Google TPU 适配说明（torch_tpu / PyTorch/XLA 之外的官方 Python 栈）

TPU 与国产卡是两类完全不同的适配工作。国产卡要补的是**算子缺口**（缺 complex STFT、
缺可用的 fused RNN、缺 grouped conv1d…），TPU 的算子面几乎是满的，要补的是
**数值契约**：TPU 上的 fp32 默认不按 fp32 算，`torch.istft` 和 `F.fold` 都对不上
参考。

本文所有数字都来自实测环境（见文末"测量环境"），不是推断。

## 支持矩阵

| | |
| --- | --- |
| `--device` | `tpu` |
| 后端包 | `torch_tpu`（不是 `torch_xla`） |
| 设备类型 | `tpu`（torch_tpu 把 PrivateUse1 改名而来） |
| torch 版本 | 2.11 – 2.15（torch_tpu 0.2.0 的硬约束） |
| Python | 3.13+（torch_tpu 0.2.0 的硬约束） |
| 实测硬件 | TPU v5e1（Colab `--tpu v5e1`），16 GiB HBM |

## 安装

```sh
# Colab / 官方 VM 镜像里 torch_tpu 与 libtpu 已自带，不需要装
python -c "import torch_tpu, torch; print(torch.accelerator.device_count())"
pip install --no-deps pymss
```

注意：`torch_tpu` 自带一份 `torch`（实测 `2.14.1+cpu`），**不要**再 `pip install torch`，
否则会把 XLA 桥接层换掉。

## 使用

```sh
pymss separate input.wav --device tpu -o out/
```

`--device auto` 在 TPU 机器上也会选到 `tpu:0`，但它是**最后一个**被探测的后端：import
`torch_tpu` 会启动 libtpu runtime，所以只有前面所有后端都不可用时才会走到它。

## 四处必须改的地方（已进 `pymss_core/device.py`）

`--device tpu` 的 profile：

```python
TPU: DeviceProfile(
    TPU, "tpu", plugin="torch_tpu", complex_fft=False, rnn=True,
    amp=True, amp_dtype=torch.bfloat16, channels_last=False, multi_device=False,
    group_norm_contiguous=True, matmul_precision="high", fold_exact=True,
    fft_result_check=False,
),
```

下面每条都对应一个实测缺口。

### 一、`torch.istft` 在 TPU 上又慢又错 —— 走 pymss 自己的 `istft_npu`

这是整个适配里影响最大的一条。

```
                          rel_max      rel_l2      耗时
torch.stft               3.1e-07      3.1e-07     0.6 ms   ← 精确
torch.istft              2.8e-03      1.5e-03   217.4 ms   ← 又慢又错
pymss istft_npu          2.1e-07      1.5e-07     6.8 ms   ← 精确，32x 快
```

`torch.istft` 的 1.5e-03 误差会**整段吞掉**模型的数值精度——mask core 本身是
1.5e-05，但整个 forward 出来是 3.2e-03，全部来自 istft。

`pymss_core/npu_fft.py` 的 `istft_npu` 是基于 `irfft` + `index_add_` 重叠相加的
纯 PyTorch 组合，本来是为 Ascend 写的，在 TPU 上**既准又快**。`complex_fft=False`
就会让 `stft_roformer` / `istft_roformer` 都走这条路。

代价是 `stft_npu` 比原生 `torch.stft` 慢 0.7 ms（0.6 → 1.3 ms），换来 istft 省 210 ms。

**整条 forward 的效果（dim=384 depth=12，1 秒立体声输入）：**

| 路径 | 端到端耗时 | rel_max |
| --- | --- | --- |
| 原生 `torch.istft` | 422 ms | 3.2e-03 |
| `complex_fft=False`（`istft_npu`） | 205 ms | 3.8e-05 |

2.1x 快、84x 准，同一个开关。（这是 1 秒音频的数据，用来隔离 ISTFT 本身的开销；
10 秒音频的实测见下方性能表。）

### 二、`F.group_norm` 拒绝非连续输入 —— 加一次 `.contiguous()`

```
RuntimeError: view(): cannot create a view of shape [2, 4, 32768]
              from the input tensor of shape [2, 64, 64, 32] and strides [...]
```

TorchTPU 的 GroupNorm 内部用 `view()` 拆 group 轴，输入一旦是非连续布局（`permute`
出来的，卷积输出的常见形态）就直接抛。SCNet 与 htdemucs 都这么喂，两个模型整段挂掉。

profile 里的 `group_norm_contiguous=True` 会装
`vendor_shims.install_group_norm_contiguous_shim()`，在 kernel 前复制一次。做一次
tensor 拷贝，换来精确结果（rel 5.1e-08）。

### 三、fp32 在 TPU 上是 bf16 —— `matmul_precision="high"`

TPU 原生 **fp32 matmul 和 convolution 都跑在 bfloat16**（`Precision.DEFAULT`）。

```
                        rel（对 float64 参考）    2048³ GEMM    4096³ GEMM
DEFAULT（=bf16）              2.4e-03           25.6 TFLOPS   49.7 TFLOPS
HIGH（3x bf16 累加）          1.4e-05           27.9 TFLOPS   50.2 TFLOPS
HIGHEST（真 fp32）            2.4e-07           18.9 TFLOPS   27.5 TFLOPS
```

`high` 在 4096³ 上比默认**还快一点**，精度高 170 倍；`highest` 慢 45%。
所以 profile 默认 `high`，由 `prepare_device_alias("tpu")` 里的
`torch.set_float32_matmul_precision("high")` 一次性设好（全局开关，不在每个 matmul 点设）。

要更高精度就自己覆盖：

```sh
PYMSS_MATMUL_PRECISION=highest pymss separate ...   # 或代码里 set_float32_matmul_precision
```

### 四、`F.fold` 用 bf16 累加 —— 换成 `scatter_add_` 组合

`torch.istft` 之下还有一层。`pymss.utils` 的整个分块叠加（每个 chunk 乘窗再 fold）
走的也是设备 kernel：

```
                              rel（对 fp32 参考）    耗时
TPU F.fold（7 个 16000 窗）      2.6e-03           99 ms
pymss scatter_add_ 组合          0                4.2 ms
```

后果是整个音轨比单次 forward 差两个数量级：单条 forward 是 3e-05，但过一遍
`demix_track` 出来是 2.6e-03，误差全在重叠相加上。

`fold_exact=True` 会装 `vendor_shims.install_fold_exact_shim()`，用
`scatter_add_` 重建 col2im。它**比厂商 kernel 快 24 倍**，因为不用离开设备。
装完后 fp32 端到端从 2.6e-03 降到 **2.2e-05**。

### 五、`torch.tpu.set_device` 是内部 API

```
RuntimeError: set_device is internal to TorchTPU and can only be called by the
PyTorch framework (torch and torch_tpu). Use torch.accelerator.set_device_index instead.
```

`torch_tpu` 把 `set_device` / `device_count` / `is_available` 全部标成 "internal to
TorchTPU" 并抛 RuntimeError，指向 `torch.accelerator`。所以：

* `pymss_core.device.set_current_vendor_device` 对 `tpu` 改走
  `torch.accelerator.set_device_index()`；
* `pymss.separator._select_private_device` 改走 `torch.accelerator.device_count()`；
* `--device auto` 的可用性检查改走 `torch.accelerator.current_accelerator()`。

## 不需要做的事（都实测过）

这是 TPU 和国产卡最大的区别：**算子面基本是满的**，不需要任何 CPU 回退。

| 能力 | TPU 实测 | 对比 |
| --- | --- | --- |
| `torch.stft` / `torch.istft` 返回 complex | 原生支持 | Ascend/BIREN 需要 `complex_fft=False` |
| complex 乘 / reshape / cat / `view_as_complex` | 全部原生精确 | BIREN 需要 4 个 shim |
| LSTM / GRU | 原生，6.0e-05 / 3.0e-05，1.2 ms | BIREN 的 fused `_VF.lstm` 错到 0.18–0.47 |
| grouped conv1d / dilated conv2d / conv_transpose1d | 全部原生 | BIREN 需要 grouped shim + dilation 限制 |
| `F.pad(mode='replicate'/'reflect')` | 原生 | BIREN 需要 `replicate_pad_bins` |
| bilinear interpolate（含 bf16） | 原生 | BIREN 的 bf16 上采样没有 kernel |
| `unfold` / `fold` / `as_strided` / `std` / `scatter_add` / `index_add` | 原生，但 `fold` 数值不对（见上）| BIREN 大多数需要 shim |
| `F.glu` | 原生 | BIREN 需要 elementwise 重写 |
| SDPA | 原生（见下） | — |

因此 profile 里除 `group_norm_contiguous` 与 `fold_exact` 之外**一个 shim 都没开**。

## 已知限制

### SDPA 的广播 mask 会静默出错

TPU 的 `F.scaled_dot_product_attention` 对**需要在 query 轴上广播**的 mask
（`(B,1,1,S)` 或 3D `(1,1,S)`）给出错误结果（rel 0.80），而全形状 `(B,H,L,S)`
和 2D `(L,S)` 是对的，`is_causal=True` 也是对的。

pymss 当前的 attention（`Attend`）不传 mask，apollo 用 `is_causal`，所以**现在没有
活的 bug**。但任何要加 mask 的改动必须传全形状或 2D mask，不能依赖广播。

### bf16 推理：接近 2x，但要认精度代价

单次 forward（10 秒音频）vs 整轨 `demix_track`（含分块叠加）：

```
                       单次 forward              整轨 demix_track
fp32 + high            1.37 s  rel 3.0e-05      2.79 s  rel 2.2e-05
bf16 autocast           708 ms  rel 1.5e-02     1.45 s   rel 7.6e-03
```

默认 `use_amp: true` 在 TPU 上会走 bf16 autocast（`amp_dtype` 已设成 bfloat16
—— fp16 在 TPU 上比 fp32 还慢，永远别用）。要最高保真就把配置里的 `use_amp` 关掉。

注意整轨数字现在是**比单次 forward 更准**的（2.2e-05 < 3.0e-05）：分块叠加在
`fold_exact` 下是精确的，多次 forward 的误差被平均掉了。修 `F.fold` 之前是反过来的
（整轨 2.6e-03，差两个数量级）。

**第一次调用会花 60+ 秒**在 JIT 编译上（`demix_track` 里 shape 多样，tracing 一次
就编一次）。第二次起是上表的数字，所以任何 benchmark 都要先跑一发预热。

### `torch.compile`：整模型可编，1.46x（fp32）/ 1.59x（bf16）

早先的记录是“整模型编译失败”，现在修掉了。挡路的不是 dynomo 本身，是两个
**不可 trace 的写法**：

1. `npu_fft._degenerate()` 里 `float(out.abs().max())`——BIREN 厂商 FFT 静默返回零
   的逐调用检测。在 TPU 上 `torch.fft.rfft` 就是厂商 kernel（实测 3.1e-07），这个
   检测永远返回 False，却要两次全量归约 + 两次 host sync，而且是 graph break。
   现在改由 `DeviceProfile.fft_result_check` 控制（BIREN 默认 `True`，TPU `False`），
   两个调用点（`stft_npu` / `istft_npu`）按 profile 跳过。
2. `RMSNorm._gamma_dtype_cache` 用 `data_ptr()` + `_version` 做键，dynamo 无法 trace。
   现在编译路径走 `torch.compiler.is_compiling()` 分支，直接 `self.gamma.to(x.dtype)`，
   XLA 会按层常量折叠；eager 路径保留原缓存，**数值完全不变**。

实测（`bs_roformer` dim384 depth12，5 s 立体声，`backend="tpu"`，`dynamic=False`）：

| | eager | 编译后 | 提速 | 编译耗时 | 与 eager fp32 的相对误差 |
| --- | --- | --- | --- | --- | --- |
| fp32 | 696.3 ms | **478.0 ms** | 1.46x | **10.6 s** | 1.2e-05 |
| bf16 autocast | 361.8 ms | **227.7 ms** | 1.59x | **87.7 s** | 1.4e-02（bf16 本身）|

端到端 eager fp32 → 编译 bf16：696.3 → 227.7 ms，**3.06x**，实时倍率 7.1x → 19.4x。

但注意一个反直觉的交叉：**eager bf16（362 ms，零启动）比编译 fp32（478 ms，10.6 s）
又快又省**。编译的唯一价值是冲到 228 ms。按 5 s 音频每 clip 换算自己的盈亏平衡点：

| 方案 | 每 clip | 省下 | 回本所需音频量 |
| --- | --- | --- | --- |
| 编译 fp32 | 480 ms | 216 ms | **~4 分钟**（一首歌就回本）|
| 编译 bf16 | 228 ms | 134 ms（对 eager bf16）| **~54 分钟**（长批量才划算）|

### 编译耗时到底花在哪

同一轮干净测量拆开看：

```
eager 首次 forward               0.7 s   （没有 JIT 成本）
torch.compile 首次调用（fp32）   10.6 s
  ├─ dynamo 前端（trace 2915 个算子）  ~9 s
  └─ XLA 后端编译                     1.9 s
换一个输入长度重编               71 s
  └─ XLA 后端                        62 s
torch.compile 首次调用（bf16）   87.7 s
```

关键事实：**成本不是追踪，是 XLA 对新 shape 的 autotuning**。shape 不变时 XLA 只花
1.9 s，shape 一变就 62 s——每个 GEMM/卷积尺寸组合都要实测挑 tiling、融合方式和
buffer 排布，而且不跨 shape 复用。`dynamic=True` 本该避免它，但在 TorchTPU 上失败。

好消息是走 `demix_track` 时 shape **本来就是固定的**：`_extract_chunk` 会把尾块用
reflect/零补到 `chunk_size`，所以长度维度恒定。实测（10 s 音频，`chunk_size`=2 s，
`overlap`=1 s → 12 块）：

| `inference.batch_size` | 模型调用 | 不同 shape |
| --- | --- | --- |
| 1（默认）| 12 | **1**：`(1,2,88200)` |
| 2 | 6 | **1**：`(2,2,88200)` |
| 5 | 3 | **2**：`(5,…)`×2 + `(2,…)`×1 |
| 7 | 2 | **2**：`(7,…)`×1 + `(5,…)`×1 |

唯一残留的变长维度是**最后一个不满的 batch**（`range(0, n_chunks, batch_size)`）。
默认 `batch_size=1` 时不存在这个问题，编译一次就命中；只有 `batch_size>1` 且块数
不是它整数倍时，才多一个 shape、多一次冷编译。要消除就三行：把末批用重复块补到
`batch_size`，跑完丢掉多出来的结果。

编译耗时还对**同进程里之前跑过什么**敏感：先跑过一轮 bf16 eager 再编 fp32，会从
10.6 s 涨到 79 s（autotune 缓存被搅乱，要重 tune）。所以实测数字波动大，别拿单次
读数当基准。

所以编译是对的，但**不能默认开**：按 shape 重编才是贵的那一项，单次 CLI 调用反而变慢。`dynamic=True` 仍然失败，`dynamic=False` 下
换输入长度只重编 1–3 ms。

子模块级别的旧数字仍然成立：mask core 197 ms → 101 ms（1.95x），编译 8–31 s。

### 单设备

`torch.accelerator.device_count() == 1`（v5e1 就是一个 chip）。pymss 的多卡数据并行
是 CUDA 专用路径，所以 `multi_device=False`。

## 实测数据

### 全模型家族数值一致性（15/15 通过）

合成输入，与 CPU fp32 参考比较，`rel_max`：

```
bs_roformer              2.80e-05    bandit_v2              1.70e-05
mel_band_roformer        3.70e-05    scnet                  1.80e-05
bs_conformer             1.20e-05    apollo                 2.50e-05
mel_band_conformer       1.20e-05    mdx23c                 3.70e-05
bs_roformer_hyperace     2.00e-05    htdemucs               1.60e-05
bandit                   1.80e-05    legacy_tasnet          5.90e-05
                                       legacy_demucs          6.90e-06
                                       legacy_v3_demucs       7.70e-06
                                       legacy_hdemucs         1.60e-05
```

同一输入重复两次的相对偏差：**全部 0.0**（比 BIREN 的 1e-08 还稳）。

上 profile 之前（原生 istft + bf16 matmul）同一批是 2.6e-03 – 6.5e-03。

整轨端到端（`demix_track`，含分块叠加）fp32 是 2.2e-05——和单次 forward 同量级。
修 `F.fold` 之前整轨是 2.6e-03。

### 性能（BS-RoFormer dim=384 depth=12 heads=6，136 M 参数，10 秒 44.1 kHz 立体声）

单块 v5e chip：

| 配置 | 耗时（预热后） | 实时倍率 | rel_max |
| --- | --- | --- | --- |
| fp32 `high`（profile 默认） | 2.79 s | 3.6x | 2.2e-05 |
| bf16 autocast | 1.45 s | 6.9x | 7.6e-03 |
| 单次 forward，fp32 `high` | 1.37 s | 7.3x | 3.0e-05 |
| 单次 forward，bf16 | 708 ms | 14.1x | 1.5e-02 |
| batch=4（fp32，单次 forward） | 1.42 s/chunk | 7.0x | — |

结论：

* **bf16 是接近 2x 的提速**（1.92x 整轨 / 1.94x 单次），但精度掉三个数量级。
* **整轨比单次 forward 慢一倍**，因为 10 秒音频切两块各自带重叠区跑。
* **批处理对 `bs_roformer` 没用**：bs=1/2/4/8 的 per-clip 耗时是 698 / 658 / 652 /
  698 ms，利用率 14.5% → 15.5% → 14.5%，等于不变。但对**调度型**家族（见下）
  完全不同：`scnet` bs=8 是 2.56x，`bandit` bs=8 是 1.78x。这条要按家族看。
* **必须预热**。首次 `demix_track` 要 60–66 s 编 JIT，之后 2.79 s。

### 算力利用率：谁在瓶颈上

峰值（v5e1 单 chip，稠密 GEMM）：bf16 155 TFLOPS；fp32 `medium` 49.7 /
`high` 50.2 / `highest` 27.5（4096³）。用 `torch.utils.flop_counter` 数出真实 FLOPs：

| 模型（5 s 音频） | GFLOP | 耗时 | TFLOPS | 占 fp32 峰值 | dispatch 算子数 | dispatch 下限 |
| --- | --- | --- | --- | --- | --- | --- |
| `bs_roformer` d384 d12 | 5075 | 699 ms | 7.25 | **14.5%** | 3028 | 138 ms（20%）|
| `bandit` emb128 | 760 | 185 ms | 4.18 | 8.3% | 1876 | 85 ms（**46%**）|
| `scnet` dplayer6 | 187 | 97 ms | 1.91 | 3.8% | 535 | 24 ms（25%）|

单算子固定成本实测：**dispatch 45.5 µs/op**（`x*1.0001` 链 200 次 vs 10 次的斜率），
一次 dtype 转换约 **200 µs**（与张量大小基本无关）。所以：

* **没有模型跑在算力天花板上**——最好的 `bs_roformer` 也只有 14.5%。
* 但也不是纯 dispatch 受限：dispatch 下限只占总耗时的 20–46%。
* 真正的墙是**每个算子的规模太小**：`bs_roformer` 平均 231 µs/op、1.5 GFLOP/op，
  已经算 6.5 TFLOPS；`scnet` 平均 0.35 GFLOP/op 就只有 1.9 TFLOPS。矩阵填不满
  256×256 的 MXU，再大的峰值也用不上。
* 三个放大利用率的旋钮都验证过有效：放大模型（16.7M → 136M 参数：4.3% → 14.4%）、
  拉长音频（1 s → 10 s：5.7% → 8.0%）、攒 batch（只对调度型家族有效）。

### 为什么 bf16 在卷积系模型上反而变慢

不是 bf16 慢，是 **autocast 的转换税**。算子级实测（fp32 → bf16）：

```
大 GEMM 4096³          2.75 ms → 1.07 ms   1.0→128.9 TFLOPS
conv2d 256ch 128×128   0.99 ms → 0.57 ms   +74%
conv1d 512ch len8192   0.90 ms → 0.55 ms   +64%
conv2d 128ch 1×345     205 µs  → 198 µs     -3%
conv2d 16ch  1×345     196 µs  → 253 µs     **-22%**
linear 345x128x512     157 µs  → 193 µs     **-18%**
```

小张量上 bf16 根本占不到便宜（甚至还慢），而 autocast 每一次转换要 **200 µs**——
比一个普通算子（45 µs）贵 4 倍。盈亏平衡点很清楚：**被加速的算子必须省下 >400 µs
才划算**。4096³ 省 1.68 ms（
赚）、大 conv 省 0.42 ms（刚回本）、345 长的 conv 只省 7 µs（**净亏**）。

对得上模型级实测：

| 家族 | 算子特征 | fp32 → autocast bf16 |
| --- | --- | --- |
| `bs_roformer` / `mel_band_roformer` | 大 GEMM、少 | 699 → 364 ms（**1.92x**）|
| `bandit` / `bandit_v2` | 64 bands × 小 conv，算子最多 | 185 → 250 ms（**0.74x**）|
| `scnet` | 多尺度小 conv | 90 → 97 ms（0.93x）|
| `legacy_hdemucs` | 小 conv + LSTM | 72 → 82 ms（0.88x）|

`bandit` 差 65 ms 完全可以用转换数解释：它 1876 个算子里约 300 个被 autocast 包上转换，
300 × 200 µs ≈ 60 ms。

所以 `use_amp` 不该全局一刀切：**attention 系开，conv/RNN 系关**。

### 主机传输

`h2d` / `d2h` 约 1.2 GB/s（256 MB 实测 224 / 212 ms）。10 秒立体声是 3.5 MB，
约 3 ms —— 不是瓶颈。但要避免每个 chunk 往返一次，pymss 的
`use_complete_fast_path`（在设备上做 OLA）在 TPU 上是正确选择。

## 调试开关
```sh
PYMSS_DEVICE_BACKEND=tpu   # 强制 vendor 检测走 TPU profile
PYMSS_FFT_FALLBACK=1       # 强制走复杂安全 STFT/ISTFT 路径（TPU 默认已是 1）
PYMSS_AMP=0                # 关 autocast
PYMSS_MATMUL_PRECISION=highest   # 覆盖 fp32 matmul 精度（默认 high）
```

`fold_exact` 没有 env 开关：它只修数值、而且比厂商 kernel 快 24 倍，没有关掉的理由。

`fft_result_check` 同理：它只为 BIREN 厂商 FFT 的“静默返回零”缺陷存在，在 TPU 上
`torch.fft.rfft` 就是厂商 kernel，检测永远返回 False，却要每次 STFT 花两次全量归约 +
两次 host sync，而且阻断 `torch.compile`。默认 `True`（保住 BIREN），TPU 置 `False`。

## 开卡验证清单

```sh
# 1. 确认是 torch_tpu 而不是 torch_xla，且 Python/torch 版本满足约束
python -c "import torch, torch_tpu; print(torch.__version__, torch.accelerator.device_count())"
python -c "import sys; assert sys.version_info >= (3,13)"

# 2. 确认后端可被识别
PYMSS_DEVICE_BACKEND=tpu python -c "
from pymss_core.device import prepare_device_alias, device_profile
d = prepare_device_alias('tpu'); print(d, device_profile(d))"

# 3. 单元测试（在 TPU 机器上跑；pytest 子进程无法初始化 PjRt）
python -m pytest tests/ -q

# 4. 真实分离 smoke（10 秒音频）
pymss separate input.wav --device tpu -o out/
```

第 3 步的坑：**TPU 只能被一个进程持有**。`pytest` 用子进程跑测试时会报
`failed to initialize the PjRt plugin: The TPU is already in use by process with pid …`。
在同一个进程里 `pytest.main()` 就没问题。

## 测量环境

```
TPU v5e1 (Colab), 16 GiB HBM, 24 CPU cores, 47 GB host RAM
torch 2.14.1+cpu, torch_tpu 0.2.0, libtpu 0.0.50, jax 0.11.2, Python 3.13
```
