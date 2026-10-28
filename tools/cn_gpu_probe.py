#!/usr/bin/env python3
"""Probe the accelerator a CloudBrain/CI container is running on.

Designed to be the startup file of an OpenI ``训练任务`` (or any batch job): it
collects the vendor software stack, resolves the pymss device profile, exercises
the operators pymss relies on (STFT/ISTFT, recurrent cells), optionally runs the
test suite, and writes everything into ``c2net_context.output_path`` so the
result can be downloaded from the task page.

Every section is isolated: a missing operator or a crashing probe is recorded and
the report is still produced.

Usage:
    python tools/cn_gpu_probe.py                    # probe + tests
    python tools/cn_gpu_probe.py --device biren
    python tools/cn_gpu_probe.py --skip-tests
    python tools/cn_gpu_probe.py --install-tests     # pip install test deps first
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

STACK_KEYWORDS = (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "vllm",
    "numpy",
    "biren",
    "supa",
    "corex",
    "iluvatar",
    "metax",
    "maca",
    "gcu",
    "tops",
    "enflame",
    "dtk",
    "das",
    "hygon",
    "mlu",
    "cambricon",
    "npu",
    "cann",
    "mindspore",
    "pymss",
)

TEST_DEPS = ("pytest", "pyyaml", "librosa", "av", "soundfile", "tqdm")

# The container only receives the code repository of the project the task was
# created from, so pymss-core is fetched from the platform's own git (fast and
# reachable from inside CloudBrain) with GitHub as a fallback.
CORE_SOURCES = (
    "git+https://openi.pcl.ac.cn/baicai1145/pymss-core.git@feat/cn-gpu-adapt",
    "git+https://github.com/baicai-1145/pymss-core.git@feat/cn-gpu-adapt",
)

SECTIONS: list[tuple[str, str]] = []


def record(name: str, body: str) -> None:
    """Append one report section."""
    SECTIONS.append((name, body.strip()))
    print(f"\n=== {name} ===\n{body.strip()}", flush=True)


def run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    """Run ``cmd`` and return ``(returncode, combined output)``."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except Exception as exc:  # noqa: BLE001 - a probe must never crash the job
        return 1, f"{type(exc).__name__}: {exc}"


def pip_install(spec: str, timeout: int = 2400) -> tuple[int, str]:
    """pip install ``spec`` from the Tsinghua mirror."""
    return run(
        [sys.executable, "-m", "pip", "install", "-q", spec, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
        timeout=timeout,
    )


def ensure_packages(repo_root: Path) -> None:
    """Make ``pymss`` and ``pymss_core`` importable inside the container."""
    lines: list[str] = []
    try:
        import pymss_core  # noqa: F401

        lines.append("pymss_core: already importable")
    except ImportError:
        for source in CORE_SOURCES:
            rc, out = pip_install(source)
            lines.append(f"pymss_core <- {source}: rc={rc}")
            if rc == 0:
                break
            lines.append(out.strip()[-600:])
    try:
        import pymss  # noqa: F401

        lines.append("pymss: already importable")
    except ImportError:
        rc, out = pip_install(f"-e {repo_root}")
        lines.append(f"pymss <- pip install -e {repo_root}: rc={rc}")
        if rc != 0:
            lines.append(out.strip()[-600:])
    record("install packages", "\n".join(lines))


def probe_environment() -> None:
    """Python / platform / torch build information."""
    lines = [
        f"python      : {sys.version.split()[0]} ({sys.executable})",
        f"platform    : {platform.platform()} / {platform.machine()}",
        f"cwd         : {Path.cwd()}",
        "env         : "
        + ", ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("PYMSS_") or k in ("C2NET_", "OMPI_COMM_WORLD_SIZE")),
    ]
    try:
        import torch

        lines += [
            f"torch       : {torch.__version__}",
            f"torch.cuda  : version={getattr(torch.version, 'cuda', None)} hip={getattr(torch.version, 'hip', None)}",
            f"cuda avail  : {torch.cuda.is_available()} count={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        ]
        if torch.cuda.is_available():
            lines.append(f"device name : {torch.cuda.get_device_name(0)}")
            try:
                props = torch.cuda.get_device_properties(0)
                lines.append(
                    f"device props: {getattr(props, 'name', '?')} mem={getattr(props, 'total_memory', 0) / 2**30:.1f}GiB "
                    f"cc={getattr(props, 'major', '?')}.{getattr(props, 'minor', '?')} multi={props.multi_processor_count}"
                )
            except Exception as exc:  # noqa: BLE001
                lines.append(f"device props: unavailable ({type(exc).__name__}: {exc})")
        for extra in ("npu", "gcu", "mlu", "supa"):
            module = getattr(torch, extra, None)
            if module is not None:
                try:
                    lines.append(f"torch.{extra}  : available={module.is_available()} count={module.device_count()}")
                except Exception as exc:  # noqa: BLE001
                    lines.append(f"torch.{extra}  : error {type(exc).__name__}: {exc}")
    except Exception:  # noqa: BLE001
        lines.append("torch       : IMPORT FAILED\n" + traceback.format_exc())
    record("environment", "\n".join(lines))


def probe_stack() -> None:
    """Installed vendor packages (adapted torch, SDK bindings, frameworks)."""
    code = (
        "import importlib.metadata as md\n"
        "seen = []\n"
        "for dist in md.distributions():\n"
        "    name = (dist.metadata.get('Name') or '').lower()\n"
        "    version = dist.version or ''\n"
        "    seen.append(f'{name} {version}')\n"
        "print('\\n'.join(sorted(seen)))\n"
    )
    rc, out = run([sys.executable, "-c", code], timeout=120)
    if rc != 0:
        rc, out = run([sys.executable, "-m", "pip", "list"], timeout=120)
    kept = [
        line
        for line in out.splitlines()
        if any(keyword in line.lower() for keyword in STACK_KEYWORDS)
    ]
    record("installed stack", "\n".join(kept) if kept else out[-2000:])


def probe_profile(device: str) -> dict:
    """Resolve the pymss device profile and vendor detection inputs."""
    info: dict[str, object] = {"requested": device}
    try:
        from pymss_core.device import detect_vendor, device_profile, fft_fallback_required, rnn_fallback_required

        import torch

        device_type = torch.device(device).type if device not in ("auto",) else ("cuda" if torch.cuda.is_available() else "cpu")
        profile = device_profile(device_type)
        info.update(
            {
                "device_type": device_type,
                "vendor": profile.vendor,
                "detected_vendor": detect_vendor(device_type),
                "profile": profile.__dict__,
                "fft_fallback": fft_fallback_required(device_type),
                "rnn_fallback": rnn_fallback_required(device_type),
            }
        )
        record("pymss device profile", json.dumps(info, indent=2, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001
        info["error"] = traceback.format_exc()
        record("pymss device profile", "FAILED\n" + info["error"])
    return info


def probe_fft(device: str) -> None:
    """STFT/ISTFT round trip through the native and the fallback path."""
    import torch

    from pymss_core.npu_fft import istft_npu, stft_npu

    target = device if device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    lines: list[str] = []
    wave = torch.randn(2, 44100, device=target if torch.device(target).type != "cpu" else "cpu")
    for label, stft_fn, istft_fn in (("native", torch.stft, torch.istft), ("fallback", stft_npu, istft_npu)):
        try:
            n_fft, hop = 2048, 512
            window = torch.hann_window(n_fft, device=wave.device, dtype=torch.float32)
            spec = stft_fn(wave.float(), n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True)
            back = istft_fn(spec, n_fft=n_fft, hop_length=hop, window=window, center=True, length=wave.shape[-1])
            error = float((back.real - wave.float()).abs().max())
            lines.append(f"{label:8s}: OK  spec={tuple(spec.shape)} max_abs_err={error:.3e}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"{label:8s}: FAIL {type(exc).__name__}: {str(exc)[:300]}")
    record("stft/istft round trip", f"device={target}\n" + "\n".join(lines))


def probe_rnn(device: str) -> None:
    """Recurrent cell forward, with and without the CPU fallback."""
    import torch
    from torch import nn

    target = device if device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    lines: list[str] = []
    model = nn.LSTM(8, 16, batch_first=True).to(target).eval()
    x = torch.randn(2, 10, 8, device=target)
    try:
        with torch.inference_mode():
            out, _ = model(x)
        lines.append(f"native  : OK out={tuple(out.shape)} dtype={out.dtype}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"native  : FAIL {type(exc).__name__}: {str(exc)[:300]}")
    try:
        from pymss_core.npu_rnn import install_rnn_cpu_fallback

        install_rnn_cpu_fallback()
        with torch.inference_mode():
            out, _ = model(x)
        lines.append(f"fallback: OK out={tuple(out.shape)} dtype={out.dtype}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"fallback: FAIL {type(exc).__name__}: {str(exc)[:300]}")
    record("recurrent forward", f"device={target}\n" + "\n".join(lines))


def probe_imports() -> None:
    """Import pymss and its backend facade."""
    lines = []
    for module in ("pymss_core", "pymss", "pymss.backends"):
        try:
            imported = __import__(module, fromlist=["*"])
            version = getattr(imported, "__version__", "?")
            lines.append(f"{module:14s}: OK version={version}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"{module:14s}: FAIL {type(exc).__name__}: {exc}")
    try:
        from pymss.backends import SUPPORTED_DEVICES

        lines.append("supported devices: " + ", ".join(SUPPORTED_DEVICES))
    except Exception:  # noqa: BLE001
        pass
    record("imports", "\n".join(lines))


def probe_tests(repo_root: Path, pytest_args: str) -> None:
    """Run the unit test suite and keep the interesting tail."""
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    cmd += pytest_args.split() if pytest_args else ["test", "-k", "not live and not server"]
    rc, out = run(cmd, timeout=3600)
    tail = [line for line in out.splitlines() if line.strip()][-60:]
    record("pytest", f"cmd: {' '.join(cmd)}\nreturncode: {rc}\n" + "\n".join(tail))


def install_test_deps() -> None:
    """Install the test-only dependencies from the Tsinghua mirror."""
    rc, out = run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *TEST_DEPS,
            "-i",
            "https://pypi.tuna.tsinghua.edu.cn/simple",
        ],
        timeout=1800,
    )
    record("install test deps", f"returncode: {rc}\n" + out[-1500:])


def resolve_output_dir() -> tuple[Path, object]:
    """Return the writable output directory and the c2net context (if any)."""
    try:
        from c2net.context import prepare

        context = prepare()
        return Path(context.output_path), context
    except Exception as exc:  # noqa: BLE001
        print(f"c2net unavailable ({type(exc).__name__}: {exc}); writing next to the repo", flush=True)
        return Path.cwd() / "output", None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("PYMSS_DEVICE_BACKEND", "auto"))
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--install-tests", action="store_true")
    parser.add_argument("--no-install", action="store_true", help="skip installing pymss/pymss-core")
    parser.add_argument("--pytest-args", default="")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    print(f"probe repo: {repo_root}", flush=True)

    probe_environment()
    if not args.no_install:
        ensure_packages(repo_root)
    probe_stack()
    profile = probe_profile(args.device)
    probe_imports()
    probe_fft(args.device)
    probe_rnn(args.device)
    if args.install_tests:
        install_test_deps()
    if not args.skip_tests:
        probe_tests(repo_root, args.pytest_args)

    output_dir, context = resolve_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(f"### {name}\n{body}" for name, body in SECTIONS)
    (output_dir / "cn_gpu_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "cn_gpu_profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\nreport written to {output_dir}", flush=True)

    if context is not None and not args.skip_upload:
        try:
            from c2net.context import upload_output

            upload_output()
            print("upload_output(): ok", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"upload_output failed: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
