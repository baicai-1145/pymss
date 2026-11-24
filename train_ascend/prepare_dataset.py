"""Unpack MUSDB18-HQ into the layout pymsst expects and verify it.

Target layout:
    /workspace/musdb18hq/train/<song>/{mixture,vocals,drums,bass,other}.wav
    /workspace/musdb18hq/test/<song>/{...}.wav
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import wave
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("/workspace/musdb18hq/raw/MUSDB18-HQ.tar.gz"))
    parser.add_argument("--dest", type=Path, default=Path("/workspace/musdb18hq"))
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    staging = args.dest / "_extract"
    staging.mkdir(parents=True, exist_ok=True)
    print(f"extracting {args.archive} ({args.archive.stat().st_size / 2**30:.2f} GiB) -> {staging}", flush=True)
    with tarfile.open(args.archive, "r:gz") as tar:
        tar.extractall(staging)
    print("extracted", flush=True)

    roots = [p for p in staging.rglob("train") if p.is_dir()]
    if not roots:
        raise SystemExit("no train/ directory found after extraction")
    src = roots[0].parent
    print(f"source root: {src}", flush=True)

    for split in ("train", "test"):
        target = args.dest / split
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(src / split), str(target))

    shutil.rmtree(staging, ignore_errors=True)
    if not args.keep_archive:
        args.archive.unlink(missing_ok=True)

    report = {}
    for split in ("train", "test"):
        split_dir = args.dest / split
        songs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        stems = sorted({f.stem for s in songs for f in s.iterdir() if f.suffix in (".wav", ".flac")})
        size = sum(f.stat().st_size for s in songs for f in s.iterdir() if f.is_file())
        sample = songs[0] / "mixture.wav"
        with wave.open(str(sample), "rb") as handle:
            info = {"channels": handle.getnchannels(), "rate": handle.getframerate(),
                    "width": handle.getsampwidth(), "frames": handle.getnframes()}
        report[split] = {
            "tracks": len(songs),
            "size_gib": round(size / 2**30, 2),
            "stems": stems,
            "first5": [s.name for s in songs[:5]],
            "sample_format": info,
        }
    total = sum(r["size_gib"] for r in report.values())
    report["total_gib"] = round(total, 2)
    print(json.dumps(report, indent=2), flush=True)

    out = Path("/workspace/.tmp/dataset_report.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
