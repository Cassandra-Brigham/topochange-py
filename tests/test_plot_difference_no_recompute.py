"""Regression test: plotting must not silently recompute the difference.

``RasterPair.plot_difference()`` used to fall through to ``difference_da()``,
which calls ``compute_difference()`` with *that method's* defaults
(skip_epoch=False, bilinear) and overwrite=True. Any options the caller had
passed to their own ``compute_difference()`` call were therefore discarded,
and the difference raster on disk was silently replaced with a different one
-- so the statistics the caller had just printed no longer described the file
that every later step reads.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

import matplotlib
matplotlib.use("Agg")

from topochange.raster import Raster
from topochange.rasterpair import RasterPair


def _write_dem(path, *, z, nodata_corner=0):
    """Synthetic 1 m DEM on a fixed grid, optionally with a nodata corner."""
    h = w = 60
    data = np.asarray(z, dtype="float32")
    if nodata_corner:
        data = data.copy()
        data[:nodata_corner, :nodata_corner] = -9999.0
    profile = dict(
        driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs="EPSG:32610", nodata=-9999.0,
        transform=from_bounds(500000, 4000000, 500000 + w, 4000000 + h, w, h),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def _surface(seed, bump=0.0):
    yy, xx = np.mgrid[0:60, 0:60]
    z = 1000.0 + 5.0 * np.sin(xx / 7.0) + 3.0 * np.cos(yy / 5.0)
    rng = np.random.default_rng(seed)
    z = z + rng.normal(0, 0.02, z.shape)
    if bump:
        z[20:40, 20:40] += bump
    return z


def _digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture
def pair_and_diff(tmp_path):
    p1 = _write_dem(tmp_path / "compare.tif", z=_surface(1), nodata_corner=4)
    p2 = _write_dem(tmp_path / "reference.tif", z=_surface(1, bump=7.5))
    pair = RasterPair(Raster.from_file(str(p1)), Raster.from_file(str(p2)))
    # No output_path: compute_difference() derives one from raster2, which is
    # the same path a recompute would derive -- so a silent recompute
    # overwrites this file. That is exactly what the notebooks do.
    res = pair.compute_difference(
        skip_epoch=True,
        interpolation_method="nearest",
        mask_edge_pixels=3,          # non-default: a recompute would use 1
        verbose=False,
    )
    out = Path(res["difference_raster_path"])
    assert out.exists()
    return pair, res, out


def test_compute_difference_records_output_path(pair_and_diff):
    pair, res, out = pair_and_diff
    assert pair._last_difference_path == res["difference_raster_path"] == str(out)


def test_reported_stats_describe_the_file_on_disk(pair_and_diff):
    pair, res, out = pair_and_diff
    with rasterio.open(out) as src:
        a = src.read(1)
    a = a[np.isfinite(a)]
    st = res["stats"]
    assert st["count_valid"] == a.size
    assert st["min"] == pytest.approx(float(a.min()), abs=1e-4)
    assert st["max"] == pytest.approx(float(a.max()), abs=1e-4)


def test_plot_difference_does_not_overwrite_the_raster(pair_and_diff):
    pair, res, out = pair_and_diff
    before_digest = _digest(out)
    before_mtime = os.path.getmtime(out)

    pair.plot_difference()          # no arguments: the case that used to recompute

    assert _digest(out) == before_digest, (
        "plot_difference() rewrote the difference raster"
    )
    assert os.path.getmtime(out) == before_mtime


def test_plot_difference_still_matches_reported_stats(pair_and_diff):
    """The whole point: stats stay valid for the file after plotting."""
    pair, res, out = pair_and_diff
    pair.plot_difference()
    with rasterio.open(out) as src:
        a = src.read(1)
    a = a[np.isfinite(a)]
    assert res["stats"]["count_valid"] == a.size
    assert res["stats"]["min"] == pytest.approx(float(a.min()), abs=1e-4)


def test_explicit_pair_still_recomputes(pair_and_diff):
    """The escape hatch: pass pair= to force a fresh computation."""
    pair, res, out = pair_and_diff
    fig = pair.plot_difference(pair=pair)
    assert fig is not None


def test_explicit_diff_path_is_honoured(pair_and_diff):
    pair, res, out = pair_and_diff
    before = _digest(out)
    fig = pair.plot_difference(diff_path=str(out))
    assert fig is not None
    assert _digest(out) == before
