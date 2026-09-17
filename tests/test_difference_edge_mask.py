"""Regression test: differencing strips the resampling edge fringe.

`RasterPair.compute_difference` intersects the two DEMs' valid masks, but the
compare DEM is resampled onto the reference grid, so a thin ring of
interpolation-contaminated pixels near the data boundary (real elevations
blended with the nodata sentinel) passes the validity test and shows up as an
edge fringe in the difference. ``mask_edge_pixels`` erodes the both-valid mask
to keep only pixels where both DEMs are genuinely valid.
"""
import inspect

import numpy as np
import pytest


def test_erode_mask_strips_boundary_and_holes():
    rp = pytest.importorskip("topochange.rasterpair")
    m = np.ones((10, 10), bool)
    assert rp._erode_mask(m, 0).sum() == 100          # no-op
    assert rp._erode_mask(m, 1).sum() == 64           # 8x8 interior
    assert rp._erode_mask(m, 2).sum() == 36           # 6x6 interior
    # an interior nodata hole grows by a 1px (8-connectivity) ring
    m2 = np.ones((7, 7), bool)
    m2[3, 3] = False
    e = rp._erode_mask(m2, 1)
    assert not e[2:5, 2:5].any()                      # 3x3 around the hole removed
    assert e.sum() == 16
    # negative/None are treated as no-op
    assert rp._erode_mask(m, -3).sum() == 100


def test_mask_edge_pixels_exposed_and_defaults_to_one():
    rp = pytest.importorskip("topochange.rasterpair")
    pcp = pytest.importorskip("topochange.pointcloudpair")
    assert inspect.signature(
        rp.RasterPair.compute_difference).parameters["mask_edge_pixels"].default == 1
    assert inspect.signature(
        pcp.PointCloudPair.compute_2d_difference).parameters["mask_edge_pixels"].default == 1
