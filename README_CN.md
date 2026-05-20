# pymss
用于音乐源分离的 Python 包。
[English](./README.md)  [简体中文]
## 安装
使用 pip 安装 `pymss` 包的示例：
```sh
pip install pymss
```
## 用法
这是一个简单的例子。
```python
from pymss import MSSeparator, get_separation_logger
# 初始化
separator = MSSeparator(
    model_type='htdemucs',
    model_path='path/to/model',
    config_path='path/to/config',
    device='cuda',
    device_ids=[0],
    output_format='wav',
    use_tta=True,
    store_dirs={
        "vocals": "./output/vocals",
        "other": None # None 或缺少此音轨将导致不输出此音轨的文件。 此示例将在 ./output/vocals 中输出人声音轨，并忽略其他（乐器）音轨。 确保键与配置文件匹配。
    },
    audio_params={"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"}, # 可以省略
    logger=get_separation_logger(), # 可以省略
    debug=False, # 可以省略
    inference_params={
        "batch_size": 4,
        "overlap_size": 512,
        "chunk_size": 1024,
        "normalize": True
    } # 可以省略
)
# 处理文件夹中的所有音频文件
separator.process_folder('path/to/input_folder')
```
### 参数
- model_type: 模型类型，例如 'htdemucs'。 必须是以下之一
    ['bs_roformer',
    'mel_band_roformer',
    'htdemucs',
    'mdx23c',
    'bandit',
    'bandit_v2',
    'scnet',
    'apollo',
    'vr']
- model_path: 模型文件路径。
- config_path: 配置文件路径。
- device: 设备类型，默认为 'auto'。 必须是以下之一 ['auto', 'cuda', 'mps', 'cpu']
- device_ids: 设备 ID 列表，默认为 [0]。
- output_format: 输出音频格式，默认为 'wav'。 必须是以下之一 ['wav', 'flac', 'mp3']
- use_tta: 是否使用 TTA（测试时增强），默认为 False。 使用 TTA 会使处理时间增加三倍，但质量会略有提高。
- store_dirs: 存储目录，可以是单个文件夹路径或带有乐器键的字典。
- audio_params: 音频参数，包括 wav_bit_depth、flac_bit_depth 和 mp3_bit_rate。 默认为 {"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"}。
- logger: Logger 实例。 默认为 pymss.get_separation_logger()
- debug: 是否启用调试模式，默认为 False。
- inference_params: 推理参数，包括 batch_size、overlap_size、chunk_size、normalize、`model_compute_dtype`、`cuda_attention_backend` 和 `cuda_triton_backend`。默认值均为 None（意味着参数取配置文件或运行时默认值）。`model_type='vr'` 支持 `batch_size`、`window_size`、`aggression`、`enable_tta`、`enable_post_process`、`post_process_threshold` 和 `high_end_process`。

### CUDA Attention 后端

RoFormer 系列模型在 CUDA 上默认使用 PyTorch 默认 SDPA 路径。在 A10G 的 `BS-Roformer-HyperACE_v2_voc` 推理路径上，它比强制 cuDNN attention 更快且更稳定。需要探测式回退时可通过 `inference_params={"cuda_attention_backend": "auto"}` 覆盖，也可以为本地实验指定具体后端。可选值为 `auto`、`default`、`flash`、`cudnn`、`efficient`、`math` 和 `xformers`。`auto` 会优先尝试 cuDNN attention，然后回退到 PyTorch memory-efficient SDPA，再回退到 PyTorch 默认 SDPA。`xformers` 是本地可选安装项，不作为必需依赖。

### CUDA Triton 融合

RoFormer 系列在 CUDA 上还默认使用 `cuda_triton_backend="auto"`。这是可选路径：如果本机未安装 Triton、不是 CUDA、dtype/shape 不支持，或 Triton kernel 运行失败，pymss 会回退到常规 Torch/cuDNN/cuBLAS 路径。可用 `inference_params={"cuda_triton_backend": "off"}` 或 `"default"` 关闭。诊断值包括 `freq_atomic_out`、`attention_gate` 和 `attention_gate_out`。

Triton kernel 不替换主要 cuBLAS GEMM，而是处理 GEMM 周围大量小算子带来的 kernel launch 和中间 tensor 写回：

| 融合区域 | 融合内容 |
|---|---|
| transformer 输入路由 | time/freq transformer block 前的 4D shape routing、residual copy 和 RMSNorm |
| rotary embedding | tensor layout 和 dtype 匹配时，对 Q/K 做 in-place RoPE 旋转 |
| attention 尾部 | attention output、sigmoid gate、output projection、bias/residual 处理，短序列下可连 RoPE 一起融合 |
| mask estimator 尾部 | final tanh、grouped linear、GLU 和 mask buffer 写回 |

A10G 上，BS-Roformer-HyperACE_v2_voc 使用 `chunk_size=160000`、`batch_size=24`、`overlap_size=0`、`cuda_attention_backend="default"`，对 `test.m4a` 311.6 秒输入做端到端分离，预热 2 次、正式运行 5 次后的实测：

| 路径 | 中位耗时 | 中位 RTFx |
|---|---:|---:|
| 完全关闭 Triton | 4.306 s | 72.37x |
| Triton auto | 2.818 s | 110.59x |
| 提升 | 1.53x | 1.53x |

减少最多的是 elementwise/layout kernel：`aten::mul`、`aten::sigmoid`、`aten::rms_norm`、小 `copy_`/`clone`，以及大量临时 tensor 分配。主要收益来自减少 kernel launch 和中间 DRAM 读写；显存峰值通常变化不大，因为大激活和 GEMM 输入仍然占主要部分。实际收益会随 chunk size、batch size、模型宽度、PyTorch/Triton 版本和 GPU 改变。

启用 CUDA AMP 时，BS-RoFormer 系列默认使用 `model_compute_dtype="auto"`，会让 Linear 权重/偏置和 RMSNorm gamma 在推理中常驻 fp16，避免反复 autocast 权重转换。需要保持 fp32 参数可设置 `inference_params={"model_compute_dtype": "off"}`；CUDA RoFormer 实验可显式使用 `"float16"`。

### RoFormer overlap 速度设置

RoFormer 系列推理可显式设置 `overlap_size=0`，减少 chunk overlap 带来的重复计算，同时保留各模型原始 chunk 长度和 batch size。它会改变 chunk 边界混合方式，所以输出不会和常见 5% overlap 设置逐 bit 一致；但它不是近似 attention，也不改模型 forward。

这个 A10G 分支保留原始 HyperACE chunk schedule 作为推荐快速路径：`chunk_size=480000`、`batch_size=2`、`overlap_size=0`。它减少 chunk overlap 带来的重复计算，同时不改模型 forward。它会改变 chunk 边界混合方式，所以输出不会和 overlap 设置逐 bit 一致。

显式开启 0% overlap 的方式：

```python
inference_params={
    "overlap_size": 0,
}
```

### RoFormer 实验性近似时间 attention

RoFormer 系列有两个显式开启、默认关闭的近似时间 attention 开关。它们会改变模型输出，所以只能作为速度/质量实验使用，目标素材上需要重新验证。

`approx_time_kv_stride` 保持 Q、残差、FFN 和输出长度全分辨率，只在选中的 time-attention 层里对 K/V 做 sample 或 avg。对 BS-Roformer-HyperACE_v2_voc 来说，它明显比直接压缩整条 time-token 序列更稳。

```python
inference_params={
    "approx_time_kv_stride": 2,
    "approx_time_kv_stride_start_layer": 4,
    "approx_time_kv_stride_every": 2,
    "approx_time_kv_stride_mode": "sample",
}
```

近似模式必须保持显式开启，并在目标 GPU 和目标素材上重新验证速度与 SDR；默认仍是精确推理。

### Apple Silicon MLX 后端

在 `device='mps'` 时，可以通过 `inference_params` 显式启用可选 MLX 完整 forward：

```python
inference_params={
    "mps_model_backend": "mlx_full",
    "mps_model_compute_dtype": "float16",
}
```

该后端需要本地安装 `mlx`，但当前不会作为 `setup.py` 必需依赖安装。默认推理仍使用 Torch 路径；缺少 MLX 或 backend 运行失败时，非 VR 模型会记录 `_pymss_mlx_full_backend_error` 并回退 Torch。

### 模型兼容性

Demucs 仅支持配置为 `model: htdemucs` 且 `htdemucs.cac: true` 的 HTDemucs checkpoint。当前无外部依赖推理路径不支持 classic `model: demucs`、`model: hdemucs` 和 non-CaC Wiener Demucs 配置。

UVR VR 可通过 `model_type='vr'` 使用，支持已适配的 UVR/VR 系列 `.pth` 权重。输出 stem 名称来自内置 VR 模型列表，例如 `Vocals`、`Instrumental`、`No Echo` 或 `Echo`。

```python
separator = MSSeparator(
    model_type='vr',
    model_path='pretrain/VR_Models/1_HP-UVR.pth',
    device='cuda',
    output_format='wav',
    store_dirs={
        "Vocals": "./output/vocals",
        "Instrumental": "./output/instrumental",
    },
    inference_params={
        "batch_size": 2,
        "window_size": 512,
        "aggression": 5,
    },
)
separator.process_folder('path/to/input_folder')
```

### Hugging Face 配置提醒
一些从 Hugging Face 或 MSST-WebUI 下载的模型配置使用 `inference.num_overlap`。当前优化后的 pymss 路径使用 `inference.overlap_size`。如果配置里只有 `num_overlap`，请手动添加 `overlap_size`，或通过 `inference_params` 传入；否则 pymss 会回退到 50% overlap，推理会慢很多。

原始 HyperACE chunk schedule 的推荐快速设置：

```yaml
audio:
  chunk_size: 480000
inference:
  batch_size: 2
  overlap_size: 0
```

### A10G 实测

测试环境为 NVIDIA A10G、PyTorch 2.8.0+cu128，关闭 TTA。主模型表使用表中列出的 A10G 设置；RoFormer 和 smoke 模型均为预热 1 次、正式运行 3 次后的 `separator.separate()` 确认结果，不包含输出文件写入。RoFormer 系列使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`cuda_attention_backend="default"`、`cuda_triton_backend="auto"` 和 `model_compute_dtype="auto"`。

| 模型 | 类型 | 最快设置 | RTFx | 1 小时音频 |
|---|---|---|---:|---:|
| BS-Roformer-HyperACE_v2_voc | bs_roformer | chunk=480000, overlap=0, batch=2, Triton auto | 93.95x | 38.3s |
| model_bs_roformer_ep_368_sdr_12.9628 | bs_roformer | chunk=480000, overlap=0, batch=2, Triton auto | 33.21x | 108.4s |
| logic_bs_roformer | bs_roformer | chunk=480000, overlap=0, batch=2, Triton auto | 72.81x | 49.4s |
| mvsep_mega_model_bs_roformer_53_stems | bs_roformer | chunk=480000, overlap=0, batch=2, Triton auto | 22.58x | 159.5s |
| mel-band-roformer-deux | mel_band_roformer | chunk=480000, overlap=0, batch=2, Triton auto | 70.34x | 51.2s |
| Mel-Band-Roformer-big | mel_band_roformer | chunk=480000, overlap=0, batch=2, Triton auto | 65.94x | 54.6s |
| model_vocals_mdx23c_sdr_10.17 | mdx23c | chunk=261120, overlap=0, batch=1 | 73.25x | 49.1s |
| scnet_checkpoint_musdb18 | scnet | chunk=485100, overlap=0, batch=8 | 191.51x | 18.8s |
| model_bandit_plus_dnr_sdr_11.47 | bandit | chunk=264600, overlap=0, batch=1 | 39.19x | 91.9s |
| checkpoint-multi_state_dict | bandit_v2 | chunk=384000, overlap=0, batch=8 | 87.60x | 41.1s |
| Apollo_LQ_MP3_restoration | apollo | chunk=132300, overlap=0, batch=4 | 29.96x | 120.1s |

`HTDemucs4` smoke 权重未计入表格，因为本地 checkpoint 文件无法读取（`PytorchStreamReader failed reading zip archive`）。`models/smoke` 下的 `model_swin_upernet_ep_56_sdr_10.6703` 当前也未计入，因为这个推理版包没有暴露对应的 `model_type`。

VR 模型使用 `window_size=512`、`aggression=5`，关闭 TTA 和后处理，先搜索 batch，再按表中 batch 预热 2 次、正式运行 3 次确认。

| VR 模型 | 最快设置 | RTFx | 1 小时音频 |
|---|---|---:|---:|
| UVR-DeNoise-Lite | batch=10 | 270.00x | 13.3s |
| Harmonic_Noise_Separation_yxlllc | batch=12 | 235.63x | 15.3s |
| MGM_HIGHEND_v4 | batch=6 | 232.91x | 15.5s |
| MGM_LOWEND_A_v4 | batch=6 | 131.64x | 27.3s |
| MGM_MAIN_v4 | batch=6 | 125.51x | 28.7s |
| 10_SP-UVR-2B-32000-1 | batch=10 | 113.87x | 31.6s |
| 11_SP-UVR-2B-32000-2 | batch=6 | 113.06x | 31.8s |
| 12_SP-UVR-3B-44100 | batch=12 | 112.00x | 32.1s |
| MGM_LOWEND_B_v4 | batch=6 | 109.21x | 33.0s |
| 15_SP-UVR-MID-44100-1 | batch=6 | 108.24x | 33.3s |
| 14_SP-UVR-4B-44100-2 | batch=4 | 107.36x | 33.5s |
| 16_SP-UVR-MID-44100-2 | batch=10 | 107.33x | 33.5s |
| 13_SP-UVR-4B-44100-1 | batch=8 | 105.81x | 34.0s |
| 5_HP-Karaoke-UVR | batch=4 | 93.09x | 38.7s |
| 2_HP-UVR | batch=4 | 91.50x | 39.3s |
| UVR-DeNoise | batch=12 | 88.52x | 40.7s |
| UVR-De-Echo-Aggressive | batch=10 | 88.12x | 40.9s |
| UVR-De-Echo-Normal | batch=12 | 87.84x | 41.0s |
| 4_HP-Vocal-UVR | batch=6 | 87.77x | 41.0s |
| 3_HP-Vocal-UVR | batch=6 | 87.35x | 41.2s |
| 1_HP-UVR | batch=2 | 86.09x | 41.8s |
| UVR-DeReverb-aufr33-jarredou_4band_v4_ms_fullband | batch=12 | 83.88x | 42.9s |
| UVR-DeEcho-DeReverb | batch=6 | 83.34x | 43.2s |
| 17_HP-Wind_Inst-UVR | batch=12 | 82.82x | 43.5s |
| 6_HP-Karaoke-UVR | batch=4 | 82.02x | 43.9s |
| UVR-BVE-4B_SN-44100-1 | batch=8 | 78.57x | 45.8s |
| 8_HP2-UVR | batch=6 | 57.70x | 62.4s |
| 9_HP2-UVR | batch=4 | 57.24x | 62.9s |
| 7_HP2-UVR | batch=8 | 55.96x | 64.3s |

## 贡献
欢迎贡献！
