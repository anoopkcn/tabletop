"""What the renderer promises: a complete grid with a stable index, captions
that separate every state, frames that reproduce from the state index alone,
and a depth image that ignores the three factors it cannot see."""

from __future__ import annotations

import numpy as np
import pytest

from tabletop import (
    COUNTS,
    FACTORS,
    N_STATES,
    all_levels,
    caption_categorical,
    caption_natural,
    caption_ordinal,
    physical,
    render,
    render_uint,
    state_index,
    state_levels,
)
from tabletop.render import SIZE_R, X_POS, mini_rows, smoke_rows, subset_rows

SAMPLE = [0, 1, 4801, 99_991, N_STATES - 1]


# -- the grid ----------------------------------------------------------------
def test_grid_size():
    assert N_STATES == 294_912 == int(np.prod(COUNTS))
    assert len(FACTORS) == len(COUNTS) == 7


def test_index_round_trip():
    for i in SAMPLE:
        assert state_index(state_levels(i)) == i


def test_all_levels_is_row_major():
    lv = all_levels()
    assert lv.shape == (N_STATES, 7)
    for i in SAMPLE:
        assert np.array_equal(lv[i], state_levels(i))


def test_physical_values_are_the_documented_ones():
    lv = np.array([0, 0, 0, 0, 0, 0, 0])
    assert physical(lv)[2] == pytest.approx(SIZE_R[0]) == pytest.approx(0.20)
    lv = np.array([0, 0, 7, 0, 7, 0, 0])
    assert physical(lv)[2] == pytest.approx(SIZE_R[7]) == pytest.approx(0.382)
    assert physical(lv)[4] == pytest.approx(X_POS[7]) == pytest.approx(0.55)


# -- captions ----------------------------------------------------------------
@pytest.mark.parametrize("fn", [caption_categorical, caption_ordinal, caption_natural])
def test_captions_are_unique_over_the_whole_grid(fn):
    caps = {fn(lv) for lv in all_levels()}
    assert len(caps) == N_STATES


def test_ordinal_caption_writes_a_level_as_a_run():
    # level v of hue, size, rotation and x is written as v + 1 repeats
    words = caption_ordinal([0, 2, 0, 5, 3, 0, 0]).split()
    assert words.count("h") == 3
    assert words.count("z") == 1
    assert words.count("r") == 6
    assert words.count("x") == 4


def test_natural_caption_reads_as_a_sentence():
    assert caption_natural([0, 3, 4, 6, 2, 1, 0]) == (
        "a medium lime green cube, rotated 180 degrees clockwise, slightly left of center, "
        "lit from the back left, on a beige table"
    )


# -- frames ------------------------------------------------------------------
def test_render_shapes_and_ranges():
    rgb, depth, jitter = render(state_levels(12_345))
    assert rgb.shape == (64, 64, 3) and depth.shape == (64, 64)
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0
    assert depth.min() > 0.0 and depth.max() <= 8.0
    assert jitter.shape == (4,) and np.abs(jitter).max() <= 1.0


def test_a_frame_is_reproducible_from_its_state_index():
    for i in [7, 4801, 200_000]:
        a = render_uint(state_levels(i), i)
        b = render_uint(state_levels(i), i)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_the_two_cameras_jitter_independently():
    """Shared jitter would fingerprint a state across the modalities, which is
    the flaw version 1 of the dataset had."""
    _, _, j = render(state_levels(31_337))
    assert not np.allclose(j[:2], j[2:])


def test_no_nuisance_means_no_jitter_and_no_noise():
    rgb_a, depth_a, j = render(state_levels(555), nuisance=False)
    rgb_b, depth_b, _ = render(state_levels(555), nuisance=False)
    assert np.array_equal(rgb_a, rgb_b) and np.array_equal(depth_a, depth_b)
    assert np.all(j == 0)


# -- what each modality can and cannot see -----------------------------------
@pytest.mark.parametrize("factor", ["hue", "light", "table"])
def test_depth_cannot_see_hue_light_or_table(factor):
    """Changing one of these changes the colour image and leaves the depth
    image bit-identical, once the camera jitter is switched off."""
    k = FACTORS.index(factor)
    base = np.array([1, 0, 5, 0, 3, 0, 0])
    other = base.copy()
    other[k] = COUNTS[k] - 1
    rgb_a, depth_a, _ = render(base, nuisance=False)
    rgb_b, depth_b, _ = render(other, nuisance=False)
    assert np.array_equal(depth_a, depth_b), f"{factor} leaked into depth"
    assert not np.allclose(rgb_a, rgb_b), f"{factor} did not change the colour image"


@pytest.mark.parametrize("shape", [0, 1, 2, 3])
def test_rotation_reaches_depth_on_every_shape(shape):
    """The knob is what carries a sphere's rotation into depth; without it the
    sphere's depth map would be invariant."""
    a = np.array([shape, 0, 6, 0, 3, 0, 0])
    b = a.copy()
    b[3] = 3
    _, depth_a, _ = render(a, nuisance=False)
    _, depth_b, _ = render(b, nuisance=False)
    assert np.abs(depth_a - depth_b).max() > 1e-3


def test_rotation_reaches_colour_on_every_shape():
    for shape in range(4):
        a = np.array([shape, 0, 6, 0, 3, 0, 0])
        b = a.copy()
        b[3] = 3
        rgb_a, _, _ = render(a, nuisance=False)
        rgb_b, _, _ = render(b, nuisance=False)
        assert np.abs(rgb_a - rgb_b).max() > 1e-2


def test_rotation_is_periodic_in_the_world():
    """Twelve steps of 30 degrees close the circle: level 0 and a full turn
    would coincide, so the code a trained encoder learns has a known period."""
    from tabletop.render import ROT_DEG

    assert len(ROT_DEG) == 12
    assert ROT_DEG[0] == 0.0 and ROT_DEG[-1] == 330.0


# -- subsets -----------------------------------------------------------------
def test_smoke_subset_is_complete_on_its_own():
    rows = smoke_rows()
    assert len(rows) == 6144
    lv = all_levels()[rows]
    assert set(lv[:, 1].tolist()) == {0, 3, 6, 9}
    assert set(lv[:, 0].tolist()) == {0, 1, 2, 3}      # every shape kept
    assert set(lv[:, 2].tolist()) == set(range(8))     # every size kept


def test_mini_subset_touches_every_factor():
    rows = mini_rows()
    assert len(rows) == 512
    lv = all_levels()[rows]
    for k in range(7):
        assert len(set(lv[:, k].tolist())) >= 2, f"{FACTORS[k]} is constant in the mini subset"


def test_subset_rows_keeps_the_grid_order():
    rows = subset_rows(shape=[2], table=[1])
    assert np.array_equal(rows, np.sort(rows))
    lv = all_levels()[rows]
    assert set(lv[:, 0].tolist()) == {2} and set(lv[:, 6].tolist()) == {1}
    assert len(rows) == N_STATES // (4 * 2)
