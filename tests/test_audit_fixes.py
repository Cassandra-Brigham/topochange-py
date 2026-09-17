"""Test suite for audit fixes in topochange codebase.

Tests the following audit findings:
1. H1 - Vertical CRS preserved during horizontal-only reprojection
2. H2 - Broader exception handling in transformer_with_epoch()
3. M2 - time_info updated after add_metadata epoch changes
4. M4 - crs_history.record_transformation_entry() called
5. M8 - Compound CRS consistency

NOTE (2026-08-13 test review): the H1, H2 and M4 regression classes were
removed; tests/test_metadata_propagation.py pins the same regressions with
stronger assertions (exact EPSG codes, call counts, warning text). The M2
epoch-range/date-string variants and M8 sequential-update tests kept here
are not covered elsewhere.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest import mock

import rasterio
from rasterio.transform import from_bounds
from pyproj import CRS

from topochange.raster import Raster
from topochange import crs_utils


# fixtures

@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory for test files."""
    return str(tmp_path)


def make_test_raster(tmpdir, epsg=32613):
    """
    Create a minimal test Raster object for testing.

    Parameters
    ----------
    tmpdir : str
        Temporary directory path
    epsg : int
        EPSG code for the CRS

    Returns
    -------
    Raster
        Raster object loaded from synthetic GeoTIFF
    """
    path = os.path.join(tmpdir, "test.tif")
    transform = from_bounds(0, 0, 100, 100, 10, 10)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=10,
        width=10,
        count=1,
        dtype='float32',
        crs=f'EPSG:{epsg}',
        transform=transform
    ) as dst:
        dst.write(np.ones((1, 10, 10), dtype='float32'))
    return Raster.from_file(path)


# test H1: Vertical CRS preserved during horizontal-only reprojection

class TestM2_TimeInfoUpdatedAfterEpoch:
    """
    Test that after calling add_metadata(epoch=2011.5), the raster's
    time_info dict has epoch=2011.5 and epoch_source='add_metadata'.
    """

    def test_time_info_epoch_single_value(self, tmp_dir):
        """
        Set epoch via add_metadata with a single decimal value.
        Verify time_info is updated correctly.
        """
        raster = make_test_raster(tmp_dir)
        epoch_value = 2011.5

        # initialize time_info if needed
        if not hasattr(raster, 'time_info') or raster.time_info is None:
            raster.time_info = {}

        # call add_metadata with epoch
        raster.add_metadata(epoch=epoch_value)

        # check time_info is updated
        assert hasattr(raster, 'time_info')
        assert raster.time_info is not None
        assert raster.time_info['epoch'] == epoch_value
        assert raster.time_info['epoch_source'] == 'add_metadata'

    def test_time_info_epoch_with_range(self, tmp_dir):
        """
        Set epoch via add_metadata with a range (start, end).
        Verify time_info stores the midpoint.
        """
        raster = make_test_raster(tmp_dir)
        epoch_start = 2010.0
        epoch_end = 2012.0

        # call add_metadata with epoch range
        raster.add_metadata(epoch=(epoch_start, epoch_end))

        # check time_info
        assert hasattr(raster, 'time_info')
        assert raster.time_info is not None
        # midpoint of (2010, 2012) is 2011
        expected_midpoint = 0.5 * (epoch_start + epoch_end)
        assert raster.time_info['epoch'] == expected_midpoint
        assert raster.time_info['epoch_source'] == 'add_metadata'

    def test_time_info_epoch_with_string(self, tmp_dir):
        """
        Set epoch via add_metadata with a date string.
        Verify time_info is updated.
        """
        raster = make_test_raster(tmp_dir)

        # call add_metadata with epoch string (should parse to decimal year)
        raster.add_metadata(epoch="2011-06-15")

        # check time_info is set and has epoch_source
        assert hasattr(raster, 'time_info')
        assert raster.time_info is not None
        assert 'epoch' in raster.time_info
        assert raster.time_info['epoch_source'] == 'add_metadata'
        # epoch should be reasonable (around 2011)
        assert 2011.0 <= raster.time_info['epoch'] <= 2012.0


# test M4: crs_history.record_transformation_entry() called

class TestM8_CompoundCRSConsistency:
    """
    Test that _update_current_compound_from_components() stores horizontal CRS
    as compound when no vertical exists (not set it to None).
    """

    def test_compound_stored_when_only_horizontal_exists(self, tmp_dir):
        """
        Set only horizontal CRS and verify compound is set to that
        horizontal CRS (not None).
        """
        raster = make_test_raster(tmp_dir)

        # clear any existing CRS
        raster._current_horizontal_crs = None
        raster._current_vertical_crs = None
        raster._current_compound_crs = None

        # set only horizontal
        horiz_wkt = CRS.from_epsg(32613).to_wkt()
        raster.current_horizontal_crs = horiz_wkt

        # compound should be set to the horizontal CRS
        assert raster._current_compound_crs is not None
        assert raster._current_compound_crs == horiz_wkt

    def test_compound_stored_when_only_vertical_exists(self, tmp_dir):
        """
        Set only vertical CRS and verify compound is set to that
        vertical CRS (not None).
        """
        raster = make_test_raster(tmp_dir)

        # clear existing CRS
        raster._current_horizontal_crs = None
        raster._current_vertical_crs = None
        raster._current_compound_crs = None

        # set only vertical
        vert_wkt = CRS.from_epsg(5703).to_wkt()
        raster.current_vertical_crs = vert_wkt

        # compound should be set to the vertical CRS
        assert raster._current_compound_crs is not None
        assert raster._current_compound_crs == vert_wkt

    def test_compound_created_when_both_exist(self, tmp_dir):
        """
        Set both horizontal and vertical CRS and verify a true
        compound CRS is created.
        """
        raster = make_test_raster(tmp_dir)

        # clear existing
        raster._current_horizontal_crs = None
        raster._current_vertical_crs = None
        raster._current_compound_crs = None

        # set horizontal
        horiz_wkt = CRS.from_epsg(32613).to_wkt()
        raster.current_horizontal_crs = horiz_wkt

        # then set vertical
        vert_wkt = CRS.from_epsg(5703).to_wkt()
        raster.current_vertical_crs = vert_wkt

        # compound should be set (and should not be None)
        assert raster._current_compound_crs is not None
        # compound should be different from either component alone
        # (it should be a compound or at least contain both somehow)
        # just verify it's not the raw horizontal
        assert raster._current_compound_crs != horiz_wkt

    def test_compound_not_none_after_component_updates(self, tmp_dir):
        """
        Verify that after any valid component update,
        compound is never left as None when components exist.
        """
        raster = make_test_raster(tmp_dir)

        horiz_wkt = CRS.from_epsg(32613).to_wkt()
        vert_wkt = CRS.from_epsg(5703).to_wkt()

        # set both components
        raster.current_horizontal_crs = horiz_wkt
        raster.current_vertical_crs = vert_wkt

        # verify compound is set
        assert raster._current_compound_crs is not None

        # update just horizontal
        raster.current_horizontal_crs = CRS.from_epsg(32612).to_wkt()
        assert raster._current_compound_crs is not None

        # update just vertical
        raster.current_vertical_crs = CRS.from_epsg(5702).to_wkt()
        assert raster._current_compound_crs is not None


# integration Tests

class TestIntegration:
    """
    Integration tests combining multiple fixes.
    """

    def test_m2_with_add_metadata_and_compound_crs(self, tmp_dir):
        """
        Call add_metadata with both epoch and compound CRS,
        verify both are applied correctly.
        """
        raster = make_test_raster(tmp_dir)

        horiz_crs = CRS.from_epsg(32613)
        epoch_value = 2015.5

        raster.add_metadata(
            compound_CRS=horiz_crs,
            epoch=epoch_value
        )

        # verify epoch was recorded
        assert raster.time_info is not None
        assert raster.time_info['epoch'] == epoch_value
        assert raster.time_info['epoch_source'] == 'add_metadata'

        # verify CRS was set
        assert raster._current_compound_crs is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

