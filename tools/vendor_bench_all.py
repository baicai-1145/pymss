"""One command that produces the whole performance report.

Runs, in order and on one process at a time (two processes on the card ruin every
number):

1. achievable dense matmul peak, bf16 and fp32;
2. per-family FLOPs / time / MFU, fp32 and bf16 autocast;
3. end-to-end separation wall time with PYMSS_AMP=0 and 1.

Usage:
    python tools/vendor_bench_all.py --seconds 10,60
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(script: str, args: list[str], timeout: float) -> str:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [sys.executable, "-u", os.path.join(HERE, script), *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        return (exc.stdout or "") + f"\n[{script} timed out after {timeout:.0f}s]"
    return proc.stdout + (proc.stderr if proc.returncode else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", default="10")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--skip", default="", help="comma list of: peak, models, e2e")
    parser.add_argument("--report", default=".tmp/vendor_bench_report.md")
    args = parser.parse_args()
    skip = {part.strip() for part in args.skip.split(",") if part.strip()}

    sections: list[tuple[str, str]] = []
    if "peak" not in skip:
        sections.append(("achievable matmul peak", _run("vendor_mfu.py", ["--skip-gemm=false"], args.timeout)))
    if "models" not in skip:
        for flag, label in (([], "fp32"), (["--bf16"], "bf16 autocast")):
            sections.append((f"per-family, {label}", _run("vendor_mfu.py", ["--skip-gemm", *flag], args.timeout)))
    if "e2e" not in skip:
        sections.append(("end-to-end separation", _run("vendor_bench_e2e.py", ["--seconds", args.seconds], args.timeout)))

    report = ["# Vendor performance report", ""]
    for title, body in sections:
        report.append(f"## {title}")
        report.append("```")
        report.extend(body.strip().splitlines())
        report.append("```")
        report.append("")

    text = "\n".join(report)
    print(text)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write(text)

    peaks = {m.group(1): float(m.group(2)) for m in re.finditer(r"PEAK (\w+) ([\d.]+)", text)}
    for key, value in peaks.items():
        print(f"[peak] {key}: {value:.1f} TFLOPS")
    print(f"[report] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
