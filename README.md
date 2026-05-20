# pymss

Python package for music source separation. <br>
[English]   [简体中文](./README_CN.md)

## Install

Example of using pip to install `pymss` package：

```sh
pip install pymss
```

## Usage

Here's a simple example.
```python
from pymss import MSSeparator, get_separation_logger

# init
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
        "other": None # None or missing this stem will result in no output file for this stem. This example will output the vocal's stem in ./output/vocals and ignoring the other(instrumental) stem. Making sure the key(s) match the config file.
    },
    audio_params={"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"}, # Can be omitted
    logger=get_separation_logger(), # Can be omitted
    debug=False, # Can be omitted
    inference_params={
        "batch_size": 4,
        "overlap_size": 512,
        "chunk_size": 1024,
        "normalize": True
    } # Can be omitted
)

# process all audio files in the folder
separator.process_folder('path/to/input_folder')
```

### Parameters

- model_type: The type of model, e.g., 'htdemucs'. Must be one of 
    ['bs_roformer', 
    'mel_band_roformer', 
    'htdemucs', 
    'mdx23c', 
    'bandit', 
    'bandit_v2', 
    'scnet', 
    'apollo',
    'vr']
- model_path: The path to the model file.
- config_path: The path to the configuration file.
- device: The type of device, default is 'auto'. Must be one of ['auto', 'cuda', 'mps', 'cpu']
- device_ids: List of device IDs, default is [0].
- output_format: The output audio format, default is 'wav'. Must be one of ['wav', 'flac', 'mp3']
- use_tta: Whether to use TTA, default is False. Using TTA will triple the processing time with a little bit improvement in quality.
- store_dirs: Storage directories, can be a single folder path or a dictionary with instrument keys.
- audio_params: Audio parameters including wav_bit_depth, flac_bit_depth, and mp3_bit_rate. Default is {"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"}.
- logger: Logger instance. Default is pymss.get_separation_logger()
- debug: Whether to enable debug mode, default is False.
- inference_params: Inference parameters including batch_size, overlap_size, chunk_size, normalize, `model_compute_dtype`, `cuda_attention_backend`, and `cuda_triton_backend`. Default is all None (means all params are depended on the config file or runtime defaults). For `model_type='vr'`, supported keys are `batch_size`, `window_size`, `aggression`, `enable_tta`, `enable_post_process`, `post_process_threshold`, and `high_end_process`.

### CUDA Attention Backend

RoFormer-family models default to PyTorch's default SDPA path on CUDA. On A10G this was faster and more stable than forcing cuDNN attention for the `BS-Roformer-HyperACE_v2_voc` inference path. Override with `inference_params={"cuda_attention_backend": "auto"}` if you want fallback probing, or set a specific backend for local experiments. Valid values are `auto`, `default`, `flash`, `cudnn`, `efficient`, `math`, and `xformers`. `auto` tries cuDNN attention first, then PyTorch memory-efficient SDPA, then PyTorch default SDPA. `xformers` is optional and only used if installed locally; it is not a required dependency.

### CUDA Triton Fusion

RoFormer-family models also default to `cuda_triton_backend="auto"` on CUDA. This path is optional: if Triton is not installed, the device is not CUDA, the dtype/shape is unsupported, or a Triton kernel fails, pymss falls back to the normal Torch/cuDNN/cuBLAS path. Use `inference_params={"cuda_triton_backend": "off"}` or `"default"` to disable it. Diagnostic values include `freq_atomic_out`, `attention_gate`, and `attention_gate_out`.

The Triton kernels do not replace the main cuBLAS GEMMs. They target the small operations around the GEMMs that are expensive because they launch many kernels and write intermediate tensors:

| fused area | what is fused |
|---|---|
| transformer input routing | 4D shape routing, residual copy, and RMSNorm before the time/frequency transformer blocks |
| rotary embedding | in-place RoPE rotation for Q/K when the tensor layout and dtype match |
| attention tail | attention output, sigmoid gate, output projection, bias/residual handling, and short-sequence RoPE when supported |
| mask estimator tail | final tanh, grouped linear, GLU, and writeback into the mask buffer |

Measured on NVIDIA A10G with BS-Roformer-HyperACE_v2_voc, `chunk_size=160000`, `batch_size=24`, `overlap_size=0`, `cuda_attention_backend="default"`, `test.m4a` 311.6 s input, two warmups and five measured end-to-end separation runs:

| path | median time | median RTFx |
|---|---:|---:|
| no Triton | 4.306 s | 72.37x |
| Triton auto | 2.818 s | 110.59x |
| speedup | 1.53x | 1.53x |

The biggest reduction is in elementwise/layout kernels: `aten::mul`, `aten::sigmoid`, `aten::rms_norm`, small `copy_`/`clone`, and allocator-heavy temporary tensors. The main benefit is lower launch overhead and less intermediate DRAM traffic; peak memory usually changes little because the large model activations and GEMM inputs still dominate. The exact gain depends on chunk size, batch size, model width, installed PyTorch/Triton versions, and GPU.

When CUDA AMP is enabled, BS-RoFormer-family models default to `model_compute_dtype="auto"`, which keeps Linear weights/biases and RMSNorm gamma in fp16 for inference to avoid repeated autocast weight conversion. Set `inference_params={"model_compute_dtype": "off"}` to keep parameters in fp32. Explicit `"float16"` can be used for CUDA RoFormer experiments.

### RoFormer Overlap Speed Setting

For RoFormer-family inference, `overlap_size=0` removes redundant chunk overlap work while keeping each model's original chunk length and batch size. This changes chunk-boundary blending, so outputs are not bit-identical to the common 5% overlap setting, but it is not an approximate attention or model-forward shortcut.

This A10G branch keeps the original HyperACE chunk schedule as the recommended fast path: `chunk_size=480000`, `batch_size=2`, and `overlap_size=0`. This avoids redundant overlap work while preserving the model forward path. It changes chunk-boundary blending, so outputs are not bit-identical to overlap settings.

Use the 0% overlap setting explicitly:

```python
inference_params={
    "overlap_size": 0,
}
```

### Experimental Approximate RoFormer Time Attention

RoFormer-family models include two explicit, disabled-by-default approximate time-attention switches. They change model output, so they should be treated as speed/quality experiments and validated on the target material before use.

`approx_time_kv_stride` keeps Q, residuals, FFN, and output length at full resolution, but samples or averages K/V inside selected time-attention layers. On BS-Roformer-HyperACE_v2_voc this was substantially less destructive than reducing the full time-token stream.

```python
inference_params={
    "approx_time_kv_stride": 2,
    "approx_time_kv_stride_start_layer": 4,
    "approx_time_kv_stride_every": 2,
    "approx_time_kv_stride_mode": "sample",
}
```

Keep approximate modes explicit and revalidate both speed and SDR on the target GPU and material; exact inference remains the default.

### Apple Silicon MLX Backend

On `device='mps'`, an optional full MLX forward path can be enabled explicitly:

```python
inference_params={
    "mps_model_backend": "mlx_full",
    "mps_model_compute_dtype": "float16",
}
```

This backend requires `mlx` to be installed locally, but `mlx` is not a required dependency in `setup.py`. The default path remains Torch. If MLX is missing or a non-VR backend fails, the model records `_pymss_mlx_full_backend_error` and falls back to Torch.

### Model Compatibility

Demucs support is limited to HTDemucs checkpoints whose config uses `model: htdemucs` and `htdemucs.cac: true`. Classic `model: demucs`, `model: hdemucs`, and non-CaC Wiener Demucs configs are not supported by this dependency-free inference path.

UVR VR support is available through `model_type='vr'` for the supported UVR/VR series `.pth` weights. The model output stems are read from the built-in VR model list, for example `Vocals`, `Instrumental`, `No Echo`, or `Echo`.

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

### Hugging Face Configs

Some model configs downloaded from Hugging Face or MSST-WebUI use `inference.num_overlap`. This optimized pymss path uses `inference.overlap_size` instead. If the config only has `num_overlap`, add an explicit `overlap_size` or pass it through `inference_params`; otherwise pymss falls back to 50% overlap and inference will be much slower.

Recommended fast setting for the original HyperACE chunk schedule:

```yaml
audio:
  chunk_size: 480000
inference:
  batch_size: 2
  overlap_size: 0
```

### A10G Benchmark

Measured on an NVIDIA A10G with PyTorch 2.8.0+cu128, no TTA. Main model rows use the listed A10G settings; RoFormer and smoke rows were confirmed with one warmup and three measured `separator.separate()` runs. File writes are not included. RoFormer-family rows use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `cuda_attention_backend="default"`, `cuda_triton_backend="auto"`, and `model_compute_dtype="auto"`.

| model | type | fastest setting | RTFx | 1-hour audio |
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

`HTDemucs4` smoke weights were not benchmarked because the local checkpoint file could not be read (`PytorchStreamReader failed reading zip archive`). `model_swin_upernet_ep_56_sdr_10.6703` is present under `models/smoke`, but this inference-only package does not expose a matching `model_type`.

VR models were batch-searched with `window_size=512`, `aggression=5`, TTA off and post-processing off, then confirmed with two warmups and three measured runs at the listed batch size.

| VR model | fastest setting | RTFx | 1-hour audio |
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

## Contributing
Contributions are welcome! 
