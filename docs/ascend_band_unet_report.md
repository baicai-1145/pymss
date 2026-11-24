# pymss / pymsst Ascend 950PR 环境与基线报告

主机 `840ce62cd2aa`（Ascend950PR 单卡）· 逐步追加 · 最后更新见各小节标题

---

## 1. 硬件与容器（实测）

| 项 | 值 |
|---|---|
| hostname | `840ce62cd2aa` |
| NPU | 1× `Ascend950PR`（`A310-50-C00MM304A1`，Chip V100，PCIe `0000:71:00.0`） |
| HBM | 131072 MB 总量，空闲时占 ~5281 MB（驱动基线），单卡无 HCCS/UB 对等链路 |
| driver / firmware | 25.7.rc1.10 / 9.0.0.110.220 |
| CANN | 9.2.1（`/usr/local/Ascend/cann-9.2.1`，`ASCEND_TOOLKIT_HOME` 已设） |
| 设备节点 | `/dev/davinci0`, `/dev/davinci_manager` |
| CPU 配额 | `cpu.max=3200000/100000` → 32 核（`nproc` 报 128，不可信） |
| 内存上限 | cgroup `memory.max=34359738368` → 32 GiB（`free` 报 754 GiB 是宿主机的） |
| 可写盘 | `/workspace` overlay 300 G（298 G 可用） |
| 只读盘 | `/workspace/shared_assets` = GlusterFS 51 T（`ro`，不可写） |
| 应避开 | `/dev/mapper/openeuler-docker--data` 8.7 T 已用 90% |

## 2. Python 运行时

| 解释器 | 说明 |
|---|---|
| `/usr/bin/python3` 3.10.12 | 系统 python，无第三方包 |
| `/usr/local/python3.12.16/bin/python` 3.12.16 | **本次工作所用**，CANN 9.2.1 的 python 绑定装在这里（`PYTHONPATH` 指向 CANN site-packages） |

## 3. 仓库与分支选择（只读检查后）

全部通过 GitHub SSH（`git@github.com`，`~/.ssh/config` 走 `ssh.github.com:443`）clone 到 `/workspace`：

| 路径 | 远端 | 选用分支 | 理由 |
|---|---|---|---|
| `/workspace/pymss` | `baicai-1145/pymss` | `feat/adaptive-speaker-roformer` | 最新（2026-11-16），`separator.py` 含 89 处 npu 引用（`ascend` 分支的 NPU 工作已并入且更新） |
| `/workspace/pymsst` | `baicai-1145/pymsst` | `feat/moe-aux-and-ckpt` | 含 MoE aux loss + step 级 checkpoint（训练侧唯一有按步保存的分支） |
| `/workspace/pymss-core` | `baicai-1145/pymss-core` | `feat/conv-ffn` | 最新（2026-11-19），带 `npu_conv.py` / `npu_fft.py` / `npu_rnn.py` 与 `tests/test_npu_roformer_regression.py` |

### 3.1 pymsst 两个分支的分叉（供决策）

`feat/moe-aux-and-ckpt` = `main` + 1 个提交；`feat/adaptive-speaker-roformer` = `main` + 20+ 个 speaker/Emilia 提交。
两者**互相不包含**（不是快进关系）：

- 只在 `moe-aux-and-ckpt`：`6d01513 feat: MoE aux loss + step-level checkpointing`
- 只在 `adaptive-speaker-roformer`：`670bb34, a15bbea, 322ef61, fd8cc1d, 0880c08, 0906f32, 482b4ff, f9d1575, …`（speaker 数据管线 / Emilia 语料 / colab 工具）

按默认规则取 `feat/moe-aux-and-ckpt`（用户指定的断点保存特性在它里面）。切回另一条只需
`git -C /workspace/pymsst checkout feat/adaptive-speaker-roformer`。

### 3.2 上游已有的昇腾资产

- `pymss-core@feat/conv-ffn`：NPU 版 `conv/fft/rnn` 算子 shim + STFT/iSTFT 走 `npu_fft`（`stft_npu` / `istft_npu`），`torch_npu` 无法做 complex64 dtype cast 等问题已在代码里注释说明。
- `pymss-core@ascend-kernels`：`pymss_core/npu_kernels.py`（triton-ascend 融合核 playground），见第 8 节结论。

## 4. torch / torch_npu 安装（踩坑记录）

CANN 9.2.1 + Ascend950PR 上，**必须装 CPU-only 的 torch**。

| 组合 | 结果 |
|---|---|
| `torch==2.14.0` + `torch-npu==2.14.0` | ❌ `import torch_npu` 时 C++ 层 `terminate`：`Duplicated key 'pinned_reserve_segment_size_mb' found in device-specific configuration parser hook registration` |
| `torch==2.12.0` + `torch-npu==2.12.0.post3` | ❌ 同上（与版本无关） |
| `torch==2.12.0+cpu`（download.pytorch.org/whl/cpu）+ `torch-npu==2.12.0.post3` | ✅ 见 4.1 |

根因：PyPI 默认 torch wheel 带 CUDA 构建，`libtorch` 里已经注册了 CUDA 的 allocator config parser hook；`torch_npu` 的 `libtorch_npu.so` 在 `dlopen` 阶段又注册同名 key，`c10::AcceleratorAllocatorConfig::registerDeviceConfigParserHook` 直接抛 `c10::ValueError`（C++ 层未捕获 → 进程 abort，Python 无法 try/except）。
**结论：昇腾机必须用 `+cpu` wheel，不要用 pypi 默认的 CUDA wheel，也不要让 `uv sync` 走 `pytorch-cu128` index。**

### 4.1 最终版本（待第 4 节复核后填）

### 4.1 最终版本（固定）

| 组件 | 版本 |
|---|---|
| Python | 3.12.16 (`/usr/local/python3.12.16/bin/python`) |
| torch | **2.12.0+cpu**（download.pytorch.org/whl/cpu） |
| torch_npu | **2.12.0.post3** |
| triton | 未安装（torch+cpu 不带；triton-ascend 无 cp312 wheel，见第 8 节） |
| triton-ascend | **不可安装**：PyPI 仅 cp39/cp310/cp311（3.2.0 / rc2 / rc3 / rc4 均无 cp312） |
| numpy | 2.5.4 · scipy 1.18.1 |
| CANN | 9.2.1 · driver 25.7.rc1.10 |

`npu-smi info`（空闲基线）：

```
NPU 0 | Ascend950PR | OK | 0% | HBM 5281 / 131072 MB | No running processes
```

torch 侧可见性：`torch.npu.is_available()=True`、`device_count()=1`、`get_device_name(0)=Ascend950PR_9579`。

**重要默认值**：`torch.npu.conv.allow_hf32` 默认 **True**（近似 TF32）。fp32 数值对齐必须显式关掉，
否则 conv 层相对误差会被钉在 ~4e-4（见第 7 节）。`torch.npu.matmul.allow_hf32` 默认 False。

## 5. 模型（Phase C）

实现：`/workspace/pymss/train_ascend/freq_unet.py`（`BandUNetSeparator`），结构完全按规格：

```
BandSplit(62 bands, dim=192)
 -> enc62 LocalBandBlock -> BandDown 62->32 (pad1) -> enc32 -> BandDown 32->16 -> enc16
 -> BandDown 16->8 (192/288/256 -> 128) -> 4 x [ MSC 时域扩张卷积(d=1,2,4,8) + 8带 full freq attention(RoPE, 2 heads x 64) ]
 -> BandUp 8->16 + gated skip(enc16) -> dec16 -> BandUp 16->32 + skip(enc32) -> dec32
 -> BandUp 32->62 + skip(enc62) -> dec62 -> final_norm -> MaskEstimator -> mask_to_complex_shape
```

- 数学块全部来自 `pymss_core`（`RMSNorm`、`Attention`(含 RoPE + to_gates 门控)、`BandSplit`、`MaskEstimator`、`mask_to_complex_shape`），没有重写模型数学。
- 时间轴零降采样：所有时间轴卷积 stride=1，输出固定 690 帧（`--hop 512`, 8 s @44.1k）。
- `LocalBandBlock` = RMSNorm→depthwise Conv1d(k=7,groups=dim)→Linear 残差；RMSNorm→SwiGLU(mult=2)→Linear 残差。
- `MaskEstimator` 使用 pymss_core 自带实现（`pymss_core.modules.bandit.maskestim.MaskEstimationModule` 在上游是坏的：先调用后赋值导致 `UnboundLocalError`，因此改用 BS-RoFormer 生产路径的 `bands.MaskEstimator`）。

**参数量：20,874,036（20.87 M）**，上限 21.63 M ✅，参考 20,961,844 差 **-87,808（-0.42%）**。
宽度选择由 `sweep_params.py` 扫出（`dims=(192, 288, 256, 128)`），8 带栈被规格固定为 2 heads×64=128。

**Phase C 自检**（`selfcheck.py`，8 s 全前向+反向，CPU 与 NPU 各一次）：

| 项 | CPU | NPU |
|---|---|---|
| mask 形状 | (1, 2050, 690, 2) | (1, 2050, 690, 2) |
| 波形形状 | (2, 352800) → 训练循环中 reshape 为 (1, 2, 352800) | 同 |
| 输出有限 | ✅ | ✅ |
| 无梯度参数 | 0 | 0 |
| 非有限梯度 | 0 | 0 |
| 梯度全零参数 | 0 | 0 |
| 前向 / 反向 | 0.688 s / 1.774 s | **0.255 s / 0.324 s** |
| 峰值显存 | — | 2892.5 MB |

## 6. 训练循环（Phase D）

`/workspace/pymss/train_ascend/train.py`：裸循环、单卡、无 HCCL/Lightning。

- AdamW(lr=5e-4, betas=(0.9,0.999), wd=0.01) + 线性 warmup（默认 100 步）
- 梯度裁剪 1.0 (`clip_grad_norm_`)，EMA decay 0.999（fp32 shadow）
- `--precision 32|bf16`，**fp16 被硬拒绝**（`SystemExit`）
- 按步保存 `last.ckpt`（model+EMA+Adam+step+args）与 `ema-<step>.pt`；`--resume` 完整恢复
- 数据位置确定性：`item index -> (track, offset)` 是纯函数（hash），所以 resume 只需 step
- batch：默认 24；`--accum` 可把等效 batch 保持在 24（实测 bs24 显存 42 GB / 128 GB，无需累积）

损失：复频谱 L1 + 多分辨率 STFT(512/1024/2048，SC + log-mag L1) + SI-SDR 项 + 近静音 leakage
（人声 RMS<0.005 的窗口单独计入 leakage，**不进入 SI-SDR 统计**）。

### 6.1 单步吞吐（合成数据，`bench` 只含 fwd+bwd+AdamW，不含 istft/STFT 损失）

| batch | precision | grouped bands | median step | steps/s | 峰值显存 |
|---|---|---|---|---|---|
| 1 | 32 | False | 0.076 s | 13.141 | 3086 MB |
| 1 | bf16 | False | 0.055 s | 18.111 | 3086 MB |
| 1 | bf16 | True | 0.057 s | 17.566 | 3086 MB |
| 6 | bf16 | True | 0.296 s | 3.38 | 10804 MB |
| 24 | bf16 | True | 1.156 s | 0.865 | 42029 MB |
| 24 | bf16 | False | 0.989 s | 1.011 | 42029 MB |

注意：`PYMSS_CORE_EXPERIMENTAL_TRAIN_GROUPED_BANDS=1`（grouped bmm）在 bs24 上**更慢**
（1.156 s vs 0.989 s），因此训练里保持默认关闭。

## 7. 数值一致性（Phase E）

对齐方式：同一份 `pymss_core` 代码路径，同一随机种子同一份权重，CPU fp32 vs NPU。
CUDA G4 参考在本机不存在，CPU fp32 是可迁移的等价参考（已注明）。
两边都关闭 BandSplit/MaskEstimator 的 grouped-forward，使 matmul 配方完全一致。

**关键前提**：`torch.npu.conv.allow_hf32` 默认 **True**，必须置 False，否则 fp32 只有 ~4e-4：

| 配置 | 输出 max abs | 输出 rel RMS |
|---|---|---|
| HF32 默认（conv 走近似 TF32） | 3.121e-04 | 4.113e-04 |
| **HF32 关闭（严格 fp32）** | **8.941e-07** | **9.336e-07** |
| bf16 autocast vs fp32 CPU | 6.962e-03 | 9.247e-03 |

### 7.1 fp32：NPU vs CPU（686 层逐层，最差 6 层）

| 层 | max abs | rel RMS | 参考量级 |
|---|---|---|---|
| `dec16.p2` | 8.941e-07 | 1.506e-06 | |ref|=0.087 |
| `enc16.p2` | 8.866e-07 | 1.452e-06 | |ref|=0.086 |
| `dec32.p2` | 8.345e-07 | 1.395e-06 | |ref|=0.087 |
| `blocks.0.freq.attn.to_gates` | 1.729e-06 | 1.204e-06 | |ref|=0.364 |
| `enc32.p2` | 8.345e-07 | 1.151e-06 | |ref|=0.087 |
| `blocks.0.msc.pw` | 3.099e-06 | 1.105e-06 | |ref|=0.465 |

最差层 rel RMS = 1.506e-06，中位数 7.569e-07，输出 rel RMS 9.34e-07。

### 7.2 bf16：NPU vs fp32 CPU（最差 6 层）

| 层 | max abs | rel RMS | 参考量级 |
|---|---|---|---|
| `dec16.p2` | 6.995e-03 | 1.372e-02 | |ref|=0.087 |
| `dec32.p2` | 6.947e-03 | 1.286e-02 | |ref|=0.087 |
| `dec62.p2` | 6.971e-03 | 1.152e-02 | |ref|=0.088 |
| `enc16.p2` | 6.473e-03 | 1.128e-02 | |ref|=0.086 |
| `_estimate_masks.to_freqs.0.1` | 2.727e-03 | 1.074e-02 | |ref|=0.076 |
| `_estimate_masks.to_freqs.0` | 2.727e-03 | 1.074e-02 | |ref|=0.076 |

bf16 最差层 rel RMS = 1.372e-02，中位数 8.595e-03，输出 rel RMS 9.25e-03。

### 7.3 结论

- **fp32 与 CPU/CUDA 参考路径在 1e-6 量级等价（实测 9.3e-7）**，满足"禁止改数学形式"的对齐要求；前提是关掉 HF32。
- **bf16 误差上界**：输出 rel RMS ≤ 9.3e-3、单层 ≤ 1.4e-2、max abs ≤ 7.0e-3（输出）/ 6.0e-2（单层最差）。误差主要来自 conv/linear 的 bf16 累加（`*.p2` 与 attention 投影层最大），与算子覆盖无关。
- 未做 fp16 训练（规格禁止），也未改任何模型数学。

## 8. Triton（Phase F）

**triton-ascend 在本机不可用**：PyPI 上 `triton-ascend` 只有 3.2.0 / 3.2.0rc2-rc4，且**只发布 cp39/cp310/cp311 wheel**，本环境是 CPython 3.12.16 → `pip install triton-ascend` 无法安装；`torch==2.12.0+cpu` 也不带 triton（CUDA triton 已随 CUDA wheel 移除）。探测结果：

```json
{'available': False, 'reason': "ModuleNotFoundError: No module named 'triton'", 'env': 'PYMSS_NPU_TRITON', 'triton_version': None, 'triton_ascend_version': None}
```

三个 kernel 已实现于 `train_ascend/npu_kernels_bandunet.py`（`_fused_bandsplit_linear` / `_fused_rope` / `_fused_gated_out`），带 `triton_ascend_available()` 守卫与 eager 回退，**未验证**（无法编译）。回退路径与 eager 数值完全一致（max abs = 0，见 `kernels_test.py` 输出）：

```json
[{"kernel": "bandsplit_linear", "max_abs": 0.0, "rel_rms": 0.0, "eager_ms": 0.0365, "fused_ms": 0.037, "triton_path_taken": false}, {"kernel": "attention_gated_out", "max_abs": 0.0, "rel_rms": 0.0, "eager_ms": 0.0213, "fused_ms": 0.0313, "triton_path_taken": false}, {"kernel": "rope", "max_abs": 0.0, "rel_rms": 0.0, "eager_ms": 0.0342, "fused_ms": 0.073, "triton_path_taken": false}]
```

同一硬件上上游已测过的结论（`pymss-core@ascend-kernels: pymss_core/npu_kernels.py` 文件头）：
triton-ascend 3.2.1 在 950PR 上生成**非向量化**访存 kernel，纯拷贝 0.239 ms vs `aten::clone` 0.081 ms（3x 慢）；
元素链只有在 BLOCK=16384 的大张量上才赢（2.5x）；融合 matmul 与 MatMulV3 打平（cube 核受限）。
因此上游把 triton 路径默认关闭、保留 eager aclnn。

### 8.1 8 秒推理延迟（对照 G4 = 11.691 ms）

| precision | median | mean | min | p90 | vs G4 |
|---|---|---|---|---|---|
| 32 | 23.297 ms | 23.342 ms | 23.258 ms | 23.538 ms | 1.99x |
| bf16 | 13.84 ms | 13.839 ms | 13.766 ms | 13.896 ms | 1.18x |

bf16 与 G4 同量级（1.18x），且这是**未经任何融合优化**的 eager 路径。

### 8.2 torch_npu 侧可用的替代融合（已落在代码里）

1. **band-split**：`BandSplit` 本身已是 grouped `bmm`（62 个小 Linear → 1 次 batched GEMM）。实测训练模式下强制开启 grouped 反而更慢（1.156s vs 0.989s @bs24），故保持默认。
2. **RoPE**：`pymss_core` 已用 `cached_rotary_cos_sin` + `apply_rotary_emb_fast`（cos/sin 只算一次），`fused_rope` 回退路径去掉 `rotate_half` 的 `cat`，少一次全尺寸临时量。
3. **注意力门控输出**：`fused_gated_out` 回退用 in-place `mul_`，省掉 sigmoid+unsqueeze+新分配一次全尺寸写回（`to_gates` 已在上游 `Attention.forward` 中）。

## 9. 数据（Phase B）

镜像：**ModelScope `OmniData/MUSDB18-HQ`**（不用 HuggingFace）。

- 远端只提供**无损整包**：`raw/MUSDB18-HQ.tar.gz`（22.65 GB），内部是 `musdb18hq.zip`，解压后才是 wav。
- ModelScope SDK 单流下载只有 ~0.5 MB/s（14 分钟只走了 6.5 GB），改用**分块并发 + 断点续传**（`/workspace/.tmp/pdl.py`，8 线程 ranged GET 打 CDN 直链）后 **~50 MB/s**，20 分钟拿到全量。
- 落地：`/workspace/musdb18hq`，布局即 pymsst 期望的 `train/<song>/<stem>.wav` / `test/<song>/<stem>.wav`。

| split | track 数 | 文件数 | 体积 |
|---|---|---|---|
| train | **100** | 500 | 18.80 GiB |
| test | **50** | 250 | 10.24 GiB |
| 合计 | **150** | 750 | **29.04 GiB** |

- 格式：44.1 kHz / 16-bit PCM / 2ch，样本曲目 326.0 s；stems = mixture, vocals, drums, bass, other。
- train 前 5 首：`A Classic Education - NightOwl`、`ANiMAL - Clinic A`、`ANiMAL - Easy Tiger`、`ANiMAL - Rockshow`、`Actions - Devil's Words`
- test 前 5 首：`AM Contra - Heart Peripheral`、`Al James - Schoolboy Facination`、`Angels In Amplifiers - I'm Alright`、`Actions - Devil's Words`…
- **pymsst check-musdb 校验通过**（显式 `--data-root /workspace/musdb18hq`，因为 CLI 默认根是硬编码的 `/root/pymsst/musdb18hq`）：
  `train: 100 tracks, incomplete 0, ok=True` / `test: 50 tracks, incomplete 0, ok=True`
- 已删除原始 tar.gz，磁盘占用回归 34 G / 300 G。

## 10. 100 步基线吞吐（Phase D，真实 MUSDB 数据，bs=24）

| precision | median step | steps/s | steps/min | 等效 samples/s | 等效 bs3 steps/min | 与 G4 比（样本吞吐） |
|---|---|---|---|---|---|---|
| 32-true | 3.3454 s | 0.298 | 17.9 | 7.2 | 143 | 0.77x |
| **bf16** | **2.7366 s** | **0.364** | **21.8** | **8.8** | **174** | **0.94x** |

- G4 参考 186 steps/min @ batch 3 = **9.3 samples/s**。昇腾 bs24 bf16 = 8.8 samples/s → **0.94x**，同量级。
- bf16 只比 fp32 快 1.22x，说明瓶颈不在 cube 核而在 **62 段 MaskEstimator 的小 GEMM + host 侧 STFT**（合成分块实测 fwd+bwd 只占 0.99 s，剩下的 ~1.7 s 是 istft/多分辨率 STFT 损失与 dataloader）。
- 显存：bs24 峰值 **42 GB / 128 GB**，无需梯度累积（32 GiB 主机内存也够，dataloader 4 worker）。
- **resume 已验证**：`--resume last.ckpt` 从 step 100 恢复并继续到 102，EMA/Adam/step 全部还原。
- 检查点：`last.ckpt`(335 MB) + `ema-<step>.pt`(84 MB)，按 `--save-every` 落盘。

## 11. 评测协议（Phase G）与 100 步冒烟结果

协议（冻结，写入 `/workspace/runs/eval_protocol_windows.json`，56 窗）：
test 50 首按名字排序，第 i 窗 = 第 `i % 50` 首，相对位置 `[0.25, 0.45, 0.65, 0.85][i // 50]`，
offset = clamp(total*frac, 0, total-chunk)，8 s / 352800 采样。
（第一版用"固定 offset=1 chunk"选窗，全落在器乐前奏上，中位 SI-SDR 直接 -71 dB；已废弃并记录。）

| 权重 | 中位 SI-SDR | 均值 | 按歌 bootstrap 95% CI | 人声活跃子集中位（48 窗） | 近静音窗 | leakage RMS |
|---|---|---|---|---|---|---|
| raw（100 步） | **-3.82 dB** | -7.65 dB | [-11.95, -4.36] | -3.60 dB | 8 | 0.00373 |
| EMA(0.999) | -18.51 dB | -22.81 dB | [-27.36, -19.43] | -17.59 dB | 8 | 0.00939 |

**重要观测**：EMA decay 0.999 在 100 步后仍有 `0.999^100 = 90.5%` 是初始化权重，所以
EMA 权重（-18.5 dB）远差于 raw 权重（-3.8 dB）。这不是 bug，是 EMA 冷启动；
**短跑用 raw 权重评分，长跑才用 EMA**。`evaluate.py --weights raw|ema` 已支持。
这些数字只是"协议已建立 + 管线打通"的证据，不代表质量。

## 12. 剩余阻塞点与下一步

1. **triton-ascend 无法安装（cp312）**：这是唯一需要外部条件的阻塞。三条可选路径：
   ①换 py3.11 解释器装 triton-ascend 3.2.0 后再验证 kernel；②华为出 cp312 wheel；
   ③短期放弃 triton，用 CANN 自定义算子（Ascend C）或 torch_npu 图融合。
   注意上游同硬件实测：triton-ascend 3.2.1 的访存 kernel 比 aclnn 慢 3x，融合收益本就有限。
2. **62 段 MaskEstimator 是主要瓶颈**（62 个小 GEMM + 62 次 tanh）。下一步优先做
   grouped/packed MLP（上游 `forward_packed_estimators` 已存在，训练态默认关闭）或把
   62 段合成一次 `bmm`；这一步预计能拿回 1.5–2x。
3. **host 侧 STFT/iSTFT 损失占比高**：把多分辨率 STFT 损失搬到 NPU（`stft_npu` 已可用）或
   改为在频域直接算 SC/log-mag，可省一次 istft+重 STFT。
4. **bf16 vs fp32 收益只有 1.22x**：瓶颈不在 cube 核，别指望精度换算提速。
5. 目前只跑了 100 步，**没有任何质量结论**；下一次应跑 ≥10k 步并同时评 raw/EMA。
6. `pymss_core.modules.bandit.maskestim.MaskEstimationModule` 在上游是坏的
   （`check_*` 先调用后赋值 → `UnboundLocalError`），本实现改用 `bands.MaskEstimator`；
   若要回上游应先修这个 bug。
