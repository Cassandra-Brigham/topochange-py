"""Integration tests for the unified CI-method dispatchers.

Covers the contract added 2026-07-14:

- ``GridVariogram.parameter_uncertainty(method=...)``: every method lands
  in the SAME storage (``bootstrap_param_samples`` / ``_percentiles`` +
  ``param_ci_method``), which is exactly what ``plot_variogram()`` (via its
  key-match guard) and ``summary()`` read.
- ``RegionalUncertaintyEstimator.calc_total_uncertainty(ci_method=...)``:
  every method overwrites the SAME interval attributes
  (``mean_correlated_*_min/_max/_p025/_p975``), so totals and ``summary()``
  always reflect the selected method.

Runs on a small synthetic raster (fast, no external data).
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from topochange.variogram import GridVariogram, RasterDataHandler
from topochange.uncertainty import RegionalUncertaintyEstimator


# ─── fixtures ───────────────────────────────────────────────────────


def _make_correlated_field(n=80, range_px=12.0, sill=0.04, seed=0):
    """Cheap correlated field: white noise smoothed by an FFT kernel."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, (n, n))
    y, x = np.mgrid[0:n, 0:n]
    cy = cx = n // 2
    d2 = (y - cy) ** 2 + (x - cx) ** 2
    kernel = np.exp(-d2 / (2.0 * (range_px / 3.0) ** 2))
    kernel /= kernel.sum()
    smooth = np.real(np.fft.ifft2(np.fft.fft2(white) * np.fft.fft2(
        np.fft.ifftshift(kernel))))
    smooth *= np.sqrt(sill) / smooth.std()
    return smooth


@pytest.fixture(scope="module")
def small_gv(tmp_path_factory):
    """A 3-realisation GridVariogram fitted on an 80x80 synthetic raster."""
    path = tmp_path_factory.mktemp("ci") / "field.tif"
    arr = _make_correlated_field()
    with rasterio.open(
        str(path), "w", driver="GTiff", height=80, width=80, count=1,
        dtype="float64", transform=from_origin(0, 80, 1, 1), nodata=-9999.0,
    ) as dst:
        dst.write(arr, 1)
    rdh = RasterDataHandler(str(path), "m", 1.0)
    rdh.load_raster()

    gv = GridVariogram(rdh, n_realizations=3)
    # samples_per_area = samples per (area_side x area_side) block:
    # 400 per 40 m block on an 80 m raster -> ~1600, capped at 1200.
    gv.run(
        area_side=40.0, samples_per_area=400, max_samples=1200,
        bin_width=4.0, max_lag_multiplier=1 / 2,
        model_types=["spherical", "exponential"], max_components=1,
        criterion="aicc", seed=0,
    )
    return rdh, gv


# ─── parameter_uncertainty dispatcher ───────────────────────────────


class TestParameterUncertaintyDispatcher:

    def test_invalid_method_raises(self, small_gv):
        _, gv = small_gv
        with pytest.raises(ValueError, match="Options"):
            gv.parameter_uncertainty(method="bogus")

    def test_ensemble_populates_unified_storage(self, small_gv):
        _, gv = small_gv
        pct = gv.parameter_uncertainty(method="ensemble")
        assert gv.param_ci_method == "ensemble"
        # keys must match the central model's parameter names; this is
        # exactly the plot_variogram() guard, so a match here means the
        # plotted bands come from the selected method.
        expect = set(gv.fitted_model.composite_model.param_names)
        assert set(pct.keys()) == expect
        assert set(gv.bootstrap_param_percentiles.keys()) == expect
        for stats in pct.values():
            assert stats["p16"] <= stats["p50"] <= stats["p84"]
        # run_config stored for the field-bootstrap sigma_A methods
        assert gv.run_config["bin_width"] == 4.0

    def test_resample_flows_to_same_storage(self, small_gv):
        _, gv = small_gv
        pct = gv.parameter_uncertainty(method="resample", n_realizations=4,
                                       seed=1)
        assert gv.param_ci_method == "resample"
        assert set(pct.keys()) == set(
            gv.fitted_model.composite_model.param_names)
        assert len(gv.bootstrap_param_samples) >= 2

    def test_summary_labels_selected_method(self, small_gv):
        _, gv = small_gv
        gv.parameter_uncertainty(method="ensemble")
        text = gv.summary()
        assert "source: ensemble" in text

    def test_plot_picks_up_selected_method(self, small_gv):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _, gv = small_gv
        gv.parameter_uncertainty(method="ensemble")
        gv.plot_variogram()          # must not raise; bands from storage
        fig = plt.gcf()
        title = fig.axes[-1].get_title() if fig.axes else ""
        plt.close("all")
        assert "param CI: ensemble" in title


# ─── sigma_A ci_method dispatcher ───────────────────────────────────


def _bounds_attrs(est):
    return (
        est.mean_correlated_polygon_min, est.mean_correlated_polygon_max,
        est.mean_correlated_polygon_p025, est.mean_correlated_polygon_p975,
    )


class TestSigmaACiMethodDispatcher:

    @pytest.fixture()
    def est(self, small_gv):
        rdh, gv = small_gv
        return RegionalUncertaintyEstimator(
            raster_data_handler=rdh, variogram_analysis=gv,
            area_of_interest=box(20, 20, 45, 45),
            fitted_model=gv.fitted_model,
        )

    def test_invalid_ci_method_raises(self, est):
        with pytest.raises(ValueError, match="ci_method"):
            est.calc_total_uncertainty(n_pairs=2000, seed=0,
                                       ci_method="bogus")

    def test_default_is_realization_bands(self, est):
        est.calc_total_uncertainty(n_pairs=2000, seed=0)
        assert est.ci_method_used == "realization_bands"
        assert "realization_bands" in est.summary()
        lo, hi, p025, p975 = _bounds_attrs(est)
        assert np.isfinite([lo, hi, p025, p975]).all()
        assert lo <= hi and p025 <= p975

    def test_analytic_overwrites_bounds_and_labels(self, est):
        est.calc_total_uncertainty(n_pairs=2000, seed=0)
        bands = _bounds_attrs(est)
        est.calc_total_uncertainty(
            n_pairs=2000, seed=0, ci_method="analytic",
            ci_kwargs={"n_boot": 12, "n_pairs": 4000},
        )
        assert est.ci_method_used == "analytic"
        analytic = _bounds_attrs(est)
        assert np.isfinite(analytic).all()
        assert analytic[0] < analytic[1] and analytic[2] < analytic[3]
        # 95% interval wraps the 68% one
        assert analytic[2] <= analytic[0] and analytic[3] >= analytic[1]
        # the method actually changed the numbers (engines differ)
        assert not np.allclose(analytic, bands)
        # raster bounds set too, and totals inherit them
        assert np.isfinite(est.mean_correlated_raster_min)
        assert est.total_uncertainty_polygon_min <= \
            est.total_uncertainty_polygon_max
        assert "analytic" in est.summary()
        # engine output stashed for diagnostics
        assert "polygon" in est.sigma_a_ci_result
        # side effect guard: the sigma_A engine must not silently install
        # parameter percentiles (that would hijack the plot source)
        assert getattr(est.variogram_analysis,
                       "param_ci_method", None) != "analytic"

    def test_field_single_end_to_end(self, est):
        est.calc_total_uncertainty(
            n_pairs=2000, seed=0, ci_method="field_single", B=3,
            ci_kwargs={"n_pairs": 2000},
        )
        assert est.ci_method_used == "field_single"
        lo, hi, p025, p975 = _bounds_attrs(est)
        assert np.isfinite([lo, hi, p025, p975]).all()
        assert lo < hi and p025 < p975
        # raster-scale interval came back from the same replicates
        assert "sigma_a_raster" in est.sigma_a_ci_result
        assert np.isfinite(est.mean_correlated_raster_min)
        assert "field_single" in est.summary()

    def test_field_method_requires_run_config(self, small_gv):
        rdh, gv = small_gv
        est = RegionalUncertaintyEstimator(
            raster_data_handler=rdh, variogram_analysis=gv,
            area_of_interest=box(20, 20, 45, 45),
            fitted_model=gv.fitted_model,
        )
        saved = gv.run_config
        try:
            del gv.run_config
            with pytest.raises(ValueError, match="run config"):
                est.calc_total_uncertainty(n_pairs=2000, seed=0,
                                           ci_method="field_single", B=2)
        finally:
            gv.run_config = saved
