"""Regression test: per-DEM overwrite decoupling in compute_2d_difference.

The reference DEM (dem2) is invariant across successive calls that differ only
in the compare variant (same reference cloud, dem_type and resolution -> same
output path). ``overwrite_dem2=False`` must let it be built once and cheaply
reloaded thereafter, while dem1 and the difference raster still follow the
global ``overwrite`` flag. This pins the signature contract (the heavy behaviour
is exercised by the point-cloud integration tests).
"""
import inspect

import pytest


def test_compute_2d_difference_exposes_per_dem_overwrite():
    pcp = pytest.importorskip("topochange.pointcloudpair")
    sig = inspect.signature(pcp.PointCloudPair.compute_2d_difference)
    for p in ("overwrite_dem1", "overwrite_dem2"):
        assert p in sig.parameters, f"missing new parameter {p}"
        # Default None => fall back to the global ``overwrite`` (no behaviour
        # change unless the caller opts in).
        assert sig.parameters[p].default is None
    # global overwrite is still present and defaults to False
    assert sig.parameters["overwrite"].default is False
