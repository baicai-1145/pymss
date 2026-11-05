"""Compare FP32 CPU/XPU inference with a local model and identical audio.

Run from the repository: python -m test.manual.xpu_benchmark --input music.wav
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from pymss import MSSeparator, load_audio, save_audio
from pymss.modules.vocal_remover.uvr_lib_v5 import spec_utils
from pymss.operations import default_runtime
from pymss.operations.builtins import register_builtin_operations
from pymss.separator import _select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default="bs_roformer_voc_hyperacev2")
    parser.add_argument("--output", type=Path, default=Path(".tmp/xpu-benchmark"))
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--chunk-size", type=int, default=132300)
    parser.add_argument("--overlap-size", type=int, default=22050)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.seconds <= 0 or args.threads <= 0:
        parser.error("seconds and threads must be positive")
    if not 0 <= args.overlap_size < args.chunk_size:
        parser.error("require 0 <= overlap-size < chunk-size")
    if not torch.xpu.is_available():
        parser.error("an available Intel XPU is required")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    params = dict(batch_size=1, chunk_size=args.chunk_size,
                  overlap_size=args.overlap_size, use_amp=False)
    report = {
        "platform": platform.platform(), "torch": torch.__version__,
        "xpu_available": torch.xpu.is_available(),
        "xpu_device_name": torch.xpu.get_device_name(0),
        "cpu_threads": torch.get_num_threads(), "model": args.model,
        "inference_params": params, "measurements": {},
        "output_rms": {},
        "timing": "one warmup then one timed separation; excludes loading and file IO",
    }
    logger = SimpleNamespace(debug=lambda *a: None, warning=lambda *a: None)
    report["auto_device"] = _select_device("auto", [0], logger)
    wave = np.random.default_rng(35).normal(size=(2, 4096)).astype(np.float32)
    restored = spec_utils._torch_istft(spec_utils._torch_stft(wave, 512, 128, "xpu:0"), 128, "xpu:0")
    np.testing.assert_allclose(restored, wave, atol=1e-5, rtol=1e-5)
    report["stft_istft_max_abs_error"] = float(np.max(np.abs(restored - wave)))
    paths = {}
    for device in ("cpu", "xpu"):
        separator = MSSeparator.from_model_name(args.model, device=device, inference_params=params)
        sample_rate = separator.config.audio.get("sample_rate", 44100)
        audio, _ = load_audio(str(args.input), sr=sample_rate, mono=False)
        audio = np.ascontiguousarray(audio[..., :int(args.seconds * sample_rate)])
        report["audio_seconds"] = audio.shape[-1] / sample_rate
        report["sample_rate"] = sample_rate
        report["input_pcm_sha256"] = hashlib.sha256(audio.tobytes()).hexdigest()
        separator.separate(audio, pbar=False)
        if device == "xpu":
            torch.xpu.synchronize()
        start = perf_counter()
        outputs = separator.separate(audio, pbar=False)
        if device == "xpu":
            torch.xpu.synchronize()
        elapsed = perf_counter() - start
        report["measurements"][device] = {"seconds": elapsed, "rtf": elapsed / report["audio_seconds"]}
        print(device, report["measurements"][device], flush=True)
        paths[device] = {}
        report["output_rms"][device] = {}
        for stem, data in outputs.items():
            if not np.isfinite(data).all():
                raise ValueError(f"non-finite {device} output: {stem}")
            report["output_rms"][device][stem] = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
            path = args.output / f"{device}-{stem}.wav"
            save_audio(str(path), data, sample_rate, "wav", {"wav_bit_depth": "FLOAT"})
            paths[device][stem] = str(path)
        separator.del_cache()
        del separator, outputs
        gc.collect()
        if device == "xpu":
            torch.xpu.empty_cache()
    register_builtin_operations()
    report["agreement_cpu_reference"] = {
        stem: default_runtime().run(
            "pymss.kit.sdr", inputs={"reference": path, "estimated": paths["xpu"][stem]},
            params={}, context=SimpleNamespace(progress=None),
        )["report"] for stem, path in paths["cpu"].items()
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
