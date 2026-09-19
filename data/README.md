# data/

Rendered grids live here. Only `mini.h5` is tracked in git; the rest are
ignored, because the largest is 4.9 GB and all of them are reproducible from
`tabletop/render.py` alone.

| file | states | size | make it with |
|---|---|---|---|
| `mini.h5` | 512 | 8 MB | `make mini` — **tracked in git** |
| `smoke.h5` | 6,144 | 102 MB | `make smoke` |
| `tabletop_clean.h5` | 294,912 | 364 MB | `make clean-grid` |
| `tabletop.h5` | 294,912 | 4.9 GB | `make dataset` (about six minutes on ten cores) |

Each is a complete factor grid in its own right: a subset fixes some factors to
a few levels and keeps every combination of the rest, so nothing is sampled and
no combination is missing.

After rendering or downloading, check what you got:

```bash
python ../scripts/verify.py *.h5 --against ../CHECKSUMS.txt
```

A re-rendered file is expected to match the *content* digest (the second
column) and need not match the file digest — see the note at the top of
`CHECKSUMS.txt`. The schema of every file is in the repository README, under
"File layout".
