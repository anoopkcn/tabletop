"""Reading a rendered grid back: `Tabletop` wraps one HDF5 file and hands out
frames, physical values and the three caption variants. It holds the file open
lazily, so an instance survives being forked into dataloader workers.

    ds = Tabletop("data/tabletop.h5")
    rgb, depth, levels = ds[7]              # uint8 [64,64,3], float [64,64], int8 [7]
    ds.caption(7, "natural")                # 'a medium red cube, facing the camera, ...'
    rows = ds.where(shape="sphere", rotation=0)     # state rows matching a query
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .captions_nl import caption as _caption_natural
from .render import (
    COUNTS,
    FACTORS,
    HUE_NAMES,
    LIGHT_NAMES,
    SHAPES,
    SIZE_NAMES,
    TABLE_NAMES,
    caption_categorical,
    caption_ordinal,
)

CAPTION_VARIANTS = {
    "categorical": caption_categorical,
    "ordinal": caption_ordinal,
    "natural": _caption_natural,
}

# the level names a query may use, per factor; None means "index only"
LEVEL_NAMES = {
    "shape": SHAPES,
    "hue": HUE_NAMES,
    "size": SIZE_NAMES,
    "rotation": None,
    "x": None,
    "light": LIGHT_NAMES,
    "table": TABLE_NAMES,
}


class Tabletop:
    """One rendered grid on disk. Rows are file rows, not state indices: for a
    full grid the two agree, for a subset such as `smoke.h5` they do not, and
    `state_index` maps one to the other."""

    def __init__(self, path: str | Path, cache: bool = False):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Render it with "
                f"`python -m tabletop --out {self.path} --workers 10` "
                f"(about six minutes on ten cores), or add --smoke / --mini for a subset."
            )
        self._f = None
        self._cache = cache
        with self._open() as f:
            self.labels = f["labels"][:]
            self.values = f["values"][:]
            self.jitter = f["jitter"][:]
            self.state_index = f["state_index"][:]
            self.factors = list(FACTORS)
            self.counts = tuple(COUNTS)
            self.depth_max = float(f.attrs["depth_max"])
            self.jitter_deg = float(f.attrs["jitter_deg"])
            self.noise_std = float(f.attrs["noise_std"])
            self.nuisance = bool(f.attrs.get("nuisance", self.jitter_deg > 0 or self.noise_std > 0))
        if cache:
            with self._open() as f:
                self._images = f["images"][:]
                self._depth = f["depth"][:]

    # -- file handling ------------------------------------------------------
    def _open(self):
        import h5py

        return h5py.File(self.path, "r")

    @property
    def file(self):
        """The open HDF5 handle, opened on first use in this process."""
        if self._f is None:
            self._f = self._open()
        return self._f

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        n = "with nuisance" if self.nuisance else "nuisance-free"
        return f"Tabletop({self.path.name!r}, {len(self)} states, {n})"

    # -- frames -------------------------------------------------------------
    def images(self, rows=slice(None)):
        """RGB uint8 [n, 64, 64, 3]."""
        if self._cache:
            return self._images[rows]
        return self.file["images"][rows]

    def depth(self, rows=slice(None), metres: bool = True):
        """Depth [n, 64, 64]: ray length in scene units if `metres`, else raw uint16."""
        raw = self._depth[rows] if self._cache else self.file["depth"][rows]
        return raw.astype(np.float32) * (self.depth_max / 65535.0) if metres else raw

    def __getitem__(self, row):
        """`(rgb, depth, levels)` of one row, or of a slice / index array."""
        return self.images(row), self.depth(row), self.labels[row]

    # -- captions -----------------------------------------------------------
    def caption(self, row: int, variant: str = "categorical") -> str:
        if variant not in CAPTION_VARIANTS:
            raise KeyError(f"variant must be one of {sorted(CAPTION_VARIANTS)}, got {variant!r}")
        return CAPTION_VARIANTS[variant](self.labels[row])

    def captions(self, variant: str = "categorical") -> list[str]:
        """Every caption of the file, in row order. All are unique."""
        fn = CAPTION_VARIANTS[variant]
        return [fn(lv) for lv in self.labels]

    # -- queries ------------------------------------------------------------
    def where(self, **query) -> np.ndarray:
        """Rows whose factors match: a level index, a level name, or a list of
        either.

        >>> ds.where(shape="sphere", rotation=[0, 6])
        """
        keep = np.ones(len(self), bool)
        for factor, want in query.items():
            if factor not in FACTORS:
                raise KeyError(f"unknown factor {factor!r}; known: {FACTORS}")
            k = FACTORS.index(factor)
            wanted = want if isinstance(want, (list, tuple, np.ndarray)) else [want]
            levels = [self._level(factor, v) for v in wanted]
            keep &= np.isin(self.labels[:, k], levels)
        return np.flatnonzero(keep)

    @staticmethod
    def _level(factor: str, value) -> int:
        names = LEVEL_NAMES[factor]
        if isinstance(value, str):
            if names is None:
                raise ValueError(f"{factor} has no level names; give an index")
            if value not in names:
                raise ValueError(f"{value!r} is not a {factor}; known: {names}")
            return names.index(value)
        return int(value)

    def value(self, row: int, factor: str) -> float:
        """The physical value of one factor of one row, in the units of `UNITS`."""
        return float(self.values[row, FACTORS.index(factor)])
