"""Download MUSDB18-HQ from ModelScope (no HuggingFace) into --out.

Dataset: OmniData/MUSDB18-HQ (StorageSize ~23.5 GB).
Resumable: modelscope snapshot download skips already-complete files.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="OmniData/MUSDB18-HQ")
    parser.add_argument("--out", type=Path, default=Path("/workspace/musdb18hq"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from modelscope import dataset_snapshot_download

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    local = dataset_snapshot_download(
        args.repo,
        local_dir=str(args.out),
        max_workers=args.workers,
    )
    print(f"downloaded to {local} in {time.time() - t0:.1f}s", flush=True)

    total = 0
    files = 0
    for p in Path(local).rglob("*"):
        if p.is_file():
            files += 1
            total += p.stat().st_size
    print(f"FILES={files} BYTES={total} GiB={total / 2**30:.2f}", flush=True)
    for split in ("train", "test"):
        d = Path(local) / split
        if d.is_dir():
            print(f"  {split}: tracks={len([x for x in d.iterdir() if x.is_dir()])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
