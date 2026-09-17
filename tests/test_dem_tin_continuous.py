"""Regression test: TIN DEMs default to a continuous surface.

`create_dem(interpolation="tin")` builds the grid with PDAL
``filters.faceraster``. The ``max_triangle_edge_length`` option drops any
Delaunay triangle larger than the given length (writing NoData there); a
previously hardcoded ``2 * resolution`` punched holes through sparse-ground
areas (vegetation) and produced a speckled DTM. The option is now exposed and
defaults to None, which OMITS the key so PDAL uses its Infinity default (every
triangle rasterized -> continuous). This pins the signature contract; the raster
behaviour is covered by the PDAL-backed DEM integration tests.
"""
import inspect

import pytest


def test_create_dem_exposes_max_triangle_edge_length_default_none():
    pc = pytest.importorskip("topochange.pointcloud")
    sig = inspect.signature(pc.PointCloud.create_dem)
    assert "max_triangle_edge_length" in sig.parameters
    # None => omit the cap => PDAL default (Infinity) => continuous surface.
    assert sig.parameters["max_triangle_edge_length"].default is None


def test_dem_interpolation_defaults_to_tin():
    """TIN is the default gridding method across the DEM-creation API."""
    pc = pytest.importorskip("topochange.pointcloud")
    pcp = pytest.importorskip("topochange.pointcloudpair")
    assert inspect.signature(
        pc.PointCloud.create_dem).parameters["interpolation"].default == "tin"
    for name in ("create_dem_pair", "compute_2d_difference",
                 "full_differencing_pipeline", "process_point_cloud_pair"):
        default = inspect.signature(
            getattr(pcp.PointCloudPair, name)).parameters["interpolation"].default
        assert default == "tin", f"{name} interpolation default is {default!r}, not 'tin'"


def test_idw_params_exposed_on_high_level_methods():
    """The IDW controls (power/radius/window_size) reach create_dem from the
    high-level methods: explicitly on the DEM-creating methods, and via
    **dem_kwargs on create_dem_pair."""
    pc = pytest.importorskip("topochange.pointcloud")
    pcp = pytest.importorskip("topochange.pointcloudpair")

    # create_dem is the source of truth for the defaults.
    cd = inspect.signature(pc.PointCloud.create_dem).parameters
    assert cd["power"].default == 2.0
    assert cd["radius"].default is None
    assert cd["window_size"].default is None

    # Explicit, documented params on the methods a user calls directly.
    for name in ("compute_2d_difference", "full_differencing_pipeline",
                 "process_point_cloud_pair"):
        p = inspect.signature(getattr(pcp.PointCloudPair, name)).parameters
        assert p["power"].default == 2.0, name
        assert p["radius"].default is None, name
        assert p["window_size"].default is None, name

    # create_dem_pair forwards them through its **dem_kwargs passthrough.
    cdp = inspect.signature(pcp.PointCloudPair.create_dem_pair).parameters
    assert any(q.kind is inspect.Parameter.VAR_KEYWORD for q in cdp.values())
