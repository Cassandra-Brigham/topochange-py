"""Regression tests for the 2026-07-16 code-review fixes.

Covers:
  P1-1  stable-area statistics must exclude the *input* raster's own nodata /
        non-finite pixels (StableAreaRasterizer / StableAreaAnalyzer).
  P1-2  bootstrap_uncertainty_subsample must return the full-sample SE of the
        median (true n-out-of-n bootstrap; m-out-of-n rescaled by sqrt(p)).
  P2-1  WLS information criteria must count the residual variance (K = p + 1).
  P2-2  analytic sigma_A evaluation must not mutate the caller's model.
  P2-10 model-ensemble draw must tolerate degenerate (all-zero) weights.
"""
import numpy as np
import pytest


# ── P1-2 ────────────────────────────────────────────────────────────────────
def test_p1_2_bootstrap_median_se_is_full_sample():
    from topochange.variogram import StatisticalAnalysis

    class _RDH:
        pass

    data = np.random.default_rng(0).normal(0.0, 1.0, 40_000)
    rdh = _RDH()
    rdh.data_array = data
    sa = StatisticalAnalysis(rdh)

    analytic = 1.2533 * np.std(data) / np.sqrt(len(data))   # SE(median) at full N
    se_full = sa.bootstrap_uncertainty_subsample(n_bootstrap=400, seed=1)  # prop=1.0
    se_sub = sa.bootstrap_uncertainty_subsample(
        n_bootstrap=400, subsample_proportion=0.1, seed=1
    )
    # Both must estimate the SE of the FULL-sample median (not the reduced size).
    assert se_full == pytest.approx(analytic, rel=0.15)
    assert se_sub == pytest.approx(analytic, rel=0.20)
    # ... and the sub-sample path must NOT be ~sqrt(10)x larger any more.
    assert se_sub < 2.0 * se_full
    # seed makes it reproducible
    assert sa.bootstrap_uncertainty_subsample(n_bootstrap=100, seed=3) == \
        sa.bootstrap_uncertainty_subsample(n_bootstrap=100, seed=3)


# ── P2-1 ────────────────────────────────────────────────────────────────────
def test_p2_1_information_criteria_count_variance_parameter():
    from topochange.variogram import SingleVariogram
    from topochange.composite_variogram import CompositeVariogramModel

    sv = SingleVariogram.__new__(SingleVariogram)   # bypass __init__ (no raster needed)
    lags = np.linspace(5.0, 200.0, 20)
    variogram = 2.0 * (1.0 - np.exp(-lags / 40.0))  # synthetic exponential
    counts = np.full_like(lags, 500.0)
    weights = counts / np.maximum(variogram, 1e-6) ** 2
    model = CompositeVariogramModel(["exponential"], include_nugget=True)
    res = sv._fit_single_composite_model(
        model, lags, variogram, None, weights, rng=np.random.default_rng(0)
    )
    assert res is not None
    n = len(lags)
    k = model.n_params + 1   # the fix: +1 for the estimated residual variance
    # Check the K convention directly, independent of the fitted RSS value.
    assert (res["aic"] - n * np.log(res["rss"] / n)) == pytest.approx(2 * k, rel=1e-9)
    assert (res["aicc"] - res["aic"]) == pytest.approx(
        2 * k * (k + 1) / max(n - k - 1, 1), rel=1e-9
    )


# ── P2-2 ────────────────────────────────────────────────────────────────────
def test_p2_2_sigma_a_eval_does_not_mutate_model():
    from topochange.sigma_a_ci import _sigma_a_of_params
    from topochange.composite_variogram import CompositeVariogramModel

    model = CompositeVariogramModel(["exponential"], include_nugget=True)
    fitted = np.array([2.0, 8.0, 0.5])
    model.set_params(fitted)
    h = np.linspace(1.0, 50.0, 500)
    _ = _sigma_a_of_params(model, np.array([3.5, 20.0, 0.9]), h)  # evaluate elsewhere
    assert np.allclose(model.params, fitted)   # caller's state restored


# ── P2-10 ───────────────────────────────────────────────────────────────────
def test_p2_10_draw_from_degenerate_ensemble():
    from topochange.sigma_a_ci import _draw_model_from_ensemble
    from topochange.composite_variogram import CompositeVariogramModel

    model = CompositeVariogramModel(["exponential"], include_nugget=True)
    model.set_params(np.array([1.0, 10.0, 0.1]))
    samples = np.array([[1.0, 10.0, 0.1], [1.1, 11.0, 0.12]])
    ensemble = [(model, 0.0, samples), (model, 0.0, samples)]   # all-zero weights
    drawn = _draw_model_from_ensemble(ensemble, np.random.default_rng(0))
    assert drawn is not None and drawn.params is not None
    with pytest.raises(ValueError):
        _draw_model_from_ensemble([], np.random.default_rng(0))


# ── P1-1 ────────────────────────────────────────────────────────────────────
def test_p1_1_stable_area_excludes_input_nodata(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    gpd = pytest.importorskip("geopandas")
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from topochange.stable_area_analysis import (
        StableAreaRasterizer, StableAreaAnalyzer,
    )

    H = W = 60
    rng = np.random.default_rng(0)
    data = rng.normal(0.02, 0.15, (H, W)).astype("float32")
    INPUT_NODATA = -32767.0
    data[25:31, 25:31] = INPUT_NODATA          # voids that fall inside the polygon

    transform = from_origin(0, H, 1, 1)         # north-up, 1 m pixels
    diff_path = tmp_path / "diff.tif"
    with rasterio.open(
        diff_path, "w", driver="GTiff", height=H, width=W, count=1,
        dtype="float32", crs="EPSG:32611", transform=transform, nodata=INPUT_NODATA,
    ) as dst:
        dst.write(data, 1)

    poly = box(5, 5, 55, 55)                    # covers the interior incl. the voids
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:32611")

    analyzer = StableAreaAnalyzer(StableAreaRasterizer(str(diff_path), gdf, nodata=-9999))
    df = analyzer.stats_all(str(tmp_path / "stable.tif"))

    # Pre-fix the sentinel leaked in and drove mean to ~-1e4 and std to ~1e3.
    assert abs(float(df["mean"].iloc[0])) < 1.0
    assert 0.0 < float(df["std"].iloc[0]) < 1.0
    assert np.isfinite(float(df["mean"].iloc[0]))
