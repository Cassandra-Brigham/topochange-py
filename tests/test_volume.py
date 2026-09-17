"""Tests for topochange.volume.

The core integration function ``polygon_volume`` is tested exactly against
analytic results on synthetic rasters; the ``VolumeEstimator`` bridge is
tested against a lightweight fake RegionalUncertaintyEstimator so no
variogram fitting is required.
"""

import math

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from topochange.volume import polygon_volume, VolumeEstimator, VolumeResult


RES = 1.0                     # 1 m cells -> cell_area = 1
TRANSFORM = Affine(RES, 0, 0, 0, -RES, 100)   # origin top-left at (0, 100)


def _grid(ny=100, nx=100, fill=0.0):
    return np.full((ny, nx), fill, dtype=float)


# ──────────────────────────────────────────────────────────────────────────
# polygon_volume
# ──────────────────────────────────────────────────────────────────────────

class TestPolygonVolume:
    def test_constant_offset_exact_volume(self):
        """A constant dh over a k-cell polygon gives V = k · dh exactly."""
        arr = _grid(fill=0.5)
        poly = box(10, 60, 30, 80)          # 20 x 20 cells = 400 cells
        out = polygon_volume(arr, TRANSFORM, poly, cell_area=RES**2)
        assert out["n_valid"] == 400
        assert out["v_net"] == pytest.approx(400 * 0.5)
        assert out["v_fill"] == pytest.approx(400 * 0.5)
        assert out["v_cut"] == 0.0
        assert out["mean_dh"] == pytest.approx(0.5)
        assert out["frac_valid"] == 1.0

    def test_cut_fill_partition(self):
        """cut and fill partition the net volume: net = fill − cut."""
        arr = _grid()
        arr[:50, :] = -1.0                  # top half lowers
        arr[50:, :] = +2.0                  # bottom half raises
        poly = box(0, 0, 100, 100)
        out = polygon_volume(arr, TRANSFORM, poly, cell_area=1.0)
        assert out["v_cut"] == pytest.approx(5000.0)
        assert out["v_fill"] == pytest.approx(10000.0)
        assert out["v_net"] == pytest.approx(out["v_fill"] - out["v_cut"])

    def test_nodata_handling(self):
        """NaN cells are excluded and completeness is reported."""
        arr = _grid(fill=1.0)
        arr[:, :50] = np.nan                # half the raster is nodata
        poly = box(0, 0, 100, 100)
        out = polygon_volume(arr, TRANSFORM, poly, cell_area=1.0)
        assert out["n_valid"] == 5000
        assert out["frac_valid"] == pytest.approx(0.5)
        assert out["v_net"] == pytest.approx(5000.0)

    def test_lod_masks_gross_but_not_net(self):
        """LoD masking changes v_cut_lod/v_fill_lod, never v_net/v_cut/v_fill."""
        arr = _grid()
        arr[0:10, 0:10] = 0.05              # sub-LoD noise (100 cells)
        arr[20:30, 20:30] = 1.0             # real signal (100 cells)
        poly = box(0, 0, 100, 100)
        out = polygon_volume(arr, TRANSFORM, poly, cell_area=1.0, lod=0.1)
        assert out["v_fill"] == pytest.approx(100 * 0.05 + 100 * 1.0)
        assert out["v_fill_lod"] == pytest.approx(100 * 1.0)
        assert out["v_net"] == pytest.approx(100 * 0.05 + 100 * 1.0)

    def test_polygon_outside_raster_raises(self):
        arr = _grid()
        poly = box(1000, 1000, 1100, 1100)
        with pytest.raises(ValueError, match="0 cells inside"):
            polygon_volume(arr, TRANSFORM, poly, cell_area=1.0)

    def test_all_nodata_inside_polygon_raises(self):
        arr = _grid(fill=np.nan)
        poly = box(10, 10, 20, 20)
        with pytest.raises(ValueError, match="No valid"):
            polygon_volume(arr, TRANSFORM, poly, cell_area=1.0)

    def test_band_axis_squeezed(self):
        """(1, ny, nx) arrays (rioxarray convention) are accepted."""
        arr = _grid(fill=0.25)[np.newaxis, :, :]
        poly = box(0, 0, 10, 10)
        out = polygon_volume(arr, TRANSFORM, poly, cell_area=1.0)
        assert out["v_net"] == pytest.approx(100 * 0.25)


# ──────────────────────────────────────────────────────────────────────────
# VolumeEstimator (against a fake regional estimator)
# ──────────────────────────────────────────────────────────────────────────

class _FakeDA:
    """Minimal stand-in for a rioxarray DataArray."""
    def __init__(self, arr, transform):
        self.values = arr
        self._transform = transform
        self.rio = self
    def transform(self):
        return self._transform


class _FakeRDH:
    def __init__(self, arr, transform, resolution=1.0, unit="m"):
        self.rioxarray_obj = _FakeDA(arr, transform)
        self.resolution = resolution
        self.unit = unit


class _FakeRegional:
    """Duck-typed RegionalUncertaintyEstimator with precomputed numbers."""
    def __init__(self, arr, polygon, sigma_a=0.05, sigma_uncorr=0.001,
                 sigma_bias=0.02, sigma0=0.3):
        self.raster_data_handler = _FakeRDH(arr, TRANSFORM)
        self.polygon = polygon
        self.area = float(polygon.area)
        self.total_uncertainty_polygon = math.hypot(sigma_a, sigma_uncorr)
        self.mean_correlated_polygon = sigma_a
        self.mean_correlated_polygon_min = sigma_a * 0.8
        self.mean_correlated_polygon_max = sigma_a * 1.3
        self.mean_correlated_polygon_p025 = sigma_a * 0.6
        self.mean_correlated_polygon_p975 = sigma_a * 1.6
        self.mean_uncorrelated_polygon = sigma_uncorr
        self.sigma_a_stable = sigma_bias
        self.sigma0_uncorrelated = sigma0
        self.ci_method_used = "realization_bands"


@pytest.fixture
def fake_setup():
    arr = _grid(fill=0.5)
    poly = box(10, 60, 30, 80)              # 400 cells, area 400 m²
    return arr, poly, _FakeRegional(arr, poly)


class TestVolumeEstimator:
    def test_sigma_v_is_area_times_sigma_a(self, fake_setup):
        arr, poly, fake = fake_setup
        res = VolumeEstimator(fake).compute()
        assert isinstance(res, VolumeResult)
        assert res.v_net == pytest.approx(400 * 0.5)
        # σ_V = A·sqrt(σ_corr² + σ_uncorr²)
        expect = 400 * math.hypot(0.05, 0.001)
        assert res.sigma_v == pytest.approx(expect, rel=1e-9)
        assert res.sigma_v_corr == pytest.approx(400 * 0.05)

    def test_interval_bounds_scale_linearly(self, fake_setup):
        _, _, fake = fake_setup
        res = VolumeEstimator(fake).compute()
        assert res.sigma_v_min == pytest.approx(
            math.hypot(400 * 0.05 * 0.8, 400 * 0.001))
        assert res.sigma_v_p975 == pytest.approx(
            math.hypot(400 * 0.05 * 1.6, 400 * 0.001))

    def test_bias_folding(self, fake_setup):
        _, _, fake = fake_setup
        r0 = VolumeEstimator(fake).compute(include_bias_se=False)
        r1 = VolumeEstimator(fake).compute(include_bias_se=True)
        assert r1.sigma_v > r0.sigma_v
        expect = 400 * math.sqrt(0.05**2 + 0.001**2 + 0.02**2)
        assert r1.sigma_v == pytest.approx(expect, rel=1e-9)
        # bias term always reported alongside
        assert r0.sigma_v_bias == pytest.approx(400 * 0.02)

    def test_naive_vs_fullcorr_bracket_sigma_v(self, fake_setup):
        """naive (independent) < σ_V < fully correlated, for realistic σ_A."""
        _, _, fake = fake_setup
        res = VolumeEstimator(fake).compute()
        assert res.sigma_v_naive < res.sigma_v < res.sigma_v_fullcorr

    def test_sigma_lod_string(self, fake_setup):
        _, _, fake = fake_setup
        res = VolumeEstimator(fake).compute(lod="2sigma")
        assert res.lod == pytest.approx(2 * 0.3)

    def test_threshold_net_warns_and_stashes_original(self, fake_setup):
        _, _, fake = fake_setup
        with pytest.warns(UserWarning, match="biases"):
            res = VolumeEstimator(fake).compute(lod=10.0, threshold_net=True)
        # everything is sub-LoD -> thresholded net is 0
        assert res.v_net == 0.0
        assert res.extras["v_net_unthresholded"] == pytest.approx(200.0)

    def test_polygon_area_basis_extrapolates_gaps(self):
        arr = _grid(fill=1.0)
        arr[60:80, 10:20] = np.nan          # gap: left half of the polygon
        poly = box(10, 20, 30, 40)          # in xy -> rows 60:80, cols 10:30
        fake = _FakeRegional(arr, poly)
        with pytest.warns(UserWarning, match="valid data"):
            r_valid = VolumeEstimator(fake, area_basis="valid").compute()
        with pytest.warns(UserWarning, match="valid data"):
            r_poly = VolumeEstimator(fake, area_basis="polygon").compute()
        assert r_valid.v_net == pytest.approx(200.0)     # observed cells only
        assert r_poly.v_net == pytest.approx(400.0)      # mean × full area
        assert r_poly.sigma_v > r_valid.sigma_v          # scales with A

    def test_invalid_area_basis_raises(self, fake_setup):
        _, _, fake = fake_setup
        with pytest.raises(ValueError, match="area_basis"):
            VolumeEstimator(fake, area_basis="banana")


class TestLodAndBasisInteractions:
    def test_cut_side_lod_masking(self):
        """LoD masks small-magnitude CUT cells out of v_cut_lod (hand value)."""
        arr = _grid()
        arr[0:10, 0:10] = -2.0     # 100 cells, |dh| >= lod  -> kept
        arr[0:10, 10:20] = -0.1    # 100 cells, |dh| < lod   -> masked
        arr[0:10, 20:40] = +1.0    # 200 cells fill, kept
        poly = box(0, 90, 40, 100)  # the 10x40 strip above
        out = polygon_volume(arr, TRANSFORM, poly, cell_area=1.0, lod=0.5)
        assert out["v_cut"] == pytest.approx(100 * 2.0 + 100 * 0.1)
        assert out["v_cut_lod"] == pytest.approx(100 * 2.0)
        assert out["v_fill_lod"] == pytest.approx(200 * 1.0)

    def test_threshold_net_not_discarded_by_polygon_basis(self):
        """Regression: area_basis='polygon' used to overwrite the LoD-thresholded
        net volume with the unthresholded polygon-mean extrapolation."""
        arr = _grid()
        arr[60:80, 10:20] = 2.0    # 200 cells above LoD
        arr[60:80, 20:30] = 0.1    # 200 cells below LoD
        poly = box(10, 20, 30, 40)  # exactly those 400 cells
        fake = _FakeRegional(arr, poly)
        with pytest.warns(UserWarning, match="[Tt]hreshold|net"):
            res = VolumeEstimator(fake, area_basis="polygon").compute(
                lod=0.5, threshold_net=True
            )
        # thresholded net must win: 200*2.0 (the 0.1 cells are masked)
        assert res.v_net == pytest.approx(400.0)
        # the unthresholded net is preserved for reference
        assert res.extras["v_net_unthresholded"] == pytest.approx(420.0)

    def test_polygon_basis_extrapolation_without_threshold(self):
        """Without threshold_net, polygon basis extrapolates the valid mean."""
        arr = _grid(fill=1.0)
        arr[60:80, 10:20] = np.nan  # half the polygon is nodata
        poly = box(10, 20, 30, 40)
        fake = _FakeRegional(arr, poly)
        with pytest.warns(UserWarning, match="valid data"):
            res = VolumeEstimator(fake, area_basis="polygon").compute()
        # mean over valid cells is 1.0, extrapolated over all 400 m^2
        assert res.v_net == pytest.approx(400.0)
