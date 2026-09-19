"""The reader, against the 512-state grid tracked in the repository."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tabletop import Tabletop
from tabletop.render import render_uint

MINI = Path(__file__).resolve().parent.parent / "data" / "mini.h5"
pytestmark = pytest.mark.skipif(not MINI.exists(), reason="render it with `make mini`")


@pytest.fixture(scope="module")
def ds():
    with Tabletop(MINI) as d:
        yield d


def test_it_reports_what_is_in_the_file(ds):
    assert len(ds) == 512
    assert ds.counts == (4, 12, 8, 12, 8, 4, 2)
    assert ds.depth_max == 8.0
    assert ds.nuisance and ds.jitter_deg == 1.0 and ds.noise_std == 0.015


def test_a_row_comes_back_whole(ds):
    rgb, depth, levels = ds[3]
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    assert depth.shape == (64, 64) and depth.max() <= ds.depth_max
    assert levels.shape == (7,)


def test_rows_match_the_renderer(ds):
    for row in [0, 17, 300, len(ds) - 1]:
        im, dp, _ = render_uint(ds.labels[row], int(ds.state_index[row]))
        assert np.array_equal(im, ds.images(row))
        assert np.array_equal(dp, ds.depth(row, metres=False))


def test_captions_are_unique_within_the_file(ds):
    for variant in ["categorical", "ordinal", "natural"]:
        assert len(set(ds.captions(variant))) == len(ds)


def test_unknown_caption_variant_is_refused(ds):
    with pytest.raises(KeyError):
        ds.caption(0, "sentences")


def test_where_accepts_names_and_indices(ds):
    by_name = ds.where(shape="sphere")
    by_index = ds.where(shape=1)
    assert np.array_equal(by_name, by_index)
    assert len(by_name) == len(ds) // 4
    assert set(ds.labels[by_name, 0].tolist()) == {1}


def test_where_combines_factors(ds):
    rows = ds.where(shape="cube", table="beige", rotation=[0, 6])
    lv = ds.labels[rows]
    assert set(lv[:, 0].tolist()) == {0} and set(lv[:, 6].tolist()) == {0}
    assert set(lv[:, 3].tolist()) <= {0, 6}


def test_where_rejects_nonsense(ds):
    with pytest.raises(KeyError):
        ds.where(colour="red")
    with pytest.raises(ValueError):
        ds.where(shape="pyramid")
    with pytest.raises(ValueError):
        ds.where(rotation="quarter turn")


def test_physical_values_are_readable_by_name(ds):
    row = int(ds.where(size=0)[0])
    assert ds.value(row, "size") == pytest.approx(0.20)


def test_a_missing_file_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m tabletop"):
        Tabletop(tmp_path / "nothing.h5")
