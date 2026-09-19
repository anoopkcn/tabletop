"""The tabletop dataset: one object on a table, in colour, in depth, and in words.

A complete 294,912-state factor grid rendered by a small ray caster that this
package owns, so every frame is reproducible from its state index alone.

    from tabletop import Tabletop, render, caption_categorical, caption_ordinal

    ds = Tabletop("data/tabletop.h5")
    rgb, depth, levels = ds[0]                  # uint8 [64,64,3], float [64,64], int8 [7]
    ds.caption(0)                               # 'cube red minute r0 x0 frontleft beige'

    rgb, depth, jitter = render([0, 0, 4, 0, 3, 0, 0])   # no file needed
"""

from .captions_nl import caption as caption_natural
from .loader import Tabletop
from .render import (
    COUNTS,
    DEPTH_MAX,
    FACTORS,
    HUE_NAMES,
    JITTER_DEG,
    LIGHT_NAMES,
    N_STATES,
    NOISE_STD,
    SHAPES,
    SIZE_NAMES,
    TABLE_NAMES,
    UNITS,
    VALUES,
    all_levels,
    caption_categorical,
    caption_ordinal,
    physical,
    render,
    render_uint,
    state_index,
    state_levels,
)

__version__ = "2.0.0"
DATASET_VERSION = 2

__all__ = [
    "COUNTS",
    "DATASET_VERSION",
    "DEPTH_MAX",
    "FACTORS",
    "HUE_NAMES",
    "JITTER_DEG",
    "LIGHT_NAMES",
    "NOISE_STD",
    "N_STATES",
    "SHAPES",
    "SIZE_NAMES",
    "TABLE_NAMES",
    "UNITS",
    "VALUES",
    "Tabletop",
    "all_levels",
    "caption_categorical",
    "caption_natural",
    "caption_ordinal",
    "physical",
    "render",
    "render_uint",
    "state_index",
    "state_levels",
]
