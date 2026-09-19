#!/usr/bin/env python3
"""Two digests of a rendered grid, and a check that the renderer still agrees
with the file it wrote.

A *file* digest (sha256 of the bytes on disk) tells you a download arrived
intact. It does **not** survive a re-render: HDF5 is free to lay out a file
differently between library versions, so two byte-different files can hold
identical data. The *content* digest is sha256 over the arrays themselves, in
row order, and that is what a re-render reproduces exactly.

    python scripts/verify.py data/*.h5                  # both digests per file
    python scripts/verify.py data/mini.h5 --against CHECKSUMS.txt
    python scripts/verify.py data/mini.h5 --rerender 64  # re-render 64 rows and compare
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabletop.render import render_uint

ARRAYS = ["images", "depth", "labels", "values", "jitter", "state_index"]
BLOCK = 1 << 22


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def content_digest(path: Path, rows_per_read: int = 4096) -> str:
    """sha256 over every array of the file, in row order, with each array's
    name, dtype and shape mixed in first."""
    import h5py

    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        for name in ARRAYS:
            if name not in f:
                continue
            ds = f[name]
            h.update(f"{name}|{ds.dtype.str}|{ds.shape}".encode())
            for start in range(0, len(ds), rows_per_read):
                h.update(np.ascontiguousarray(ds[start:start + rows_per_read]).tobytes())
    return h.hexdigest()


def rerender_check(path: Path, n: int, rng_seed: int = 0) -> tuple[int, int]:
    """Re-render `n` rows sampled from the file and compare them pixel for
    pixel. Returns (checked, mismatched)."""
    import h5py

    rng = np.random.default_rng(rng_seed)
    with h5py.File(path, "r") as f:
        nuisance = bool(f.attrs.get("nuisance", f.attrs["jitter_deg"] > 0 or f.attrs["noise_std"] > 0))
        rows = np.sort(rng.choice(len(f["labels"]), min(n, len(f["labels"])), replace=False))
        bad = 0
        for row in rows:
            levels = f["labels"][row]
            index = int(f["state_index"][row])
            im, dp, jt = render_uint(levels, index, nuisance=nuisance)
            same = (
                np.array_equal(im, f["images"][row])
                and np.array_equal(dp, f["depth"][row])
                and np.allclose(jt, f["jitter"][row])
            )
            if not same:
                bad += 1
                print(f"    row {row} (state {index}) differs")
    return len(rows), bad


def parse_table(path: Path) -> dict[str, dict[str, str]]:
    """`CHECKSUMS.txt` rows: <file digest>  <content digest>  <name>."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 3:
            out[parts[2]] = {"file": parts[0], "content": parts[1]}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--against", type=Path, help="a CHECKSUMS.txt to compare with")
    p.add_argument("--rerender", type=int, metavar="N", help="re-render N random rows and compare")
    p.add_argument("--quick", action="store_true", help="skip the file digest (the slow one on a 5 GB grid)")
    a = p.parse_args()
    want = parse_table(a.against) if a.against else {}
    failed = False
    for path in a.files:
        if not path.exists():
            print(f"{path}: missing")
            failed = True
            continue
        fd = "-" * 64 if a.quick else file_digest(path)
        cd = content_digest(path)
        print(f"{fd}  {cd}  {path.name}")
        if path.name in want:
            bad_here = False
            for kind, got in [("file", fd), ("content", cd)]:
                if a.quick and kind == "file":
                    continue
                exp = want[path.name][kind]
                if got != exp:
                    print(f"  {kind} digest MISMATCH: expected {exp}")
                    bad_here = True
            print("  matches CHECKSUMS.txt" if not bad_here else "  DOES NOT match CHECKSUMS.txt")
            failed |= bad_here
        if a.rerender:
            checked, bad = rerender_check(path, a.rerender)
            print(f"  re-rendered {checked} rows, {bad} differ")
            failed |= bad > 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
