"""Tests for topochange.sigma_map: binned σ estimation, export, validation.

The organising idea of this suite is that a σ model is only trustworthy if it
recovers a σ law we *planted*.  Most tests therefore synthesise a difference
field from a known ``sigma_true(predictors)``, fit, and assert the fit comes
back within a stated tolerance, rather than asserting that the code returns
whatever it currently happens to return.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from topochange.heteroscedastic import (
    build_predictor_frame,
    evaluate_sigma_raster,
    nmad,
    qq_stats,
    standardize,
)
from topochange.sigma_map import (
    BinnedSigmaModel,
    CallableSigmaModel,
    trimmed_nmad_scale,
    uniform_edges,
    calibration_by_bin,
    cross_validate_sigma_models,
    fit_binned_sigma_model,
    misregistration_diagnostic,
    mixture_nmad,
    nd_binning,
    patch_validation,
    plot_1d_binning,
    plot_2d_binning,
    plot_sigma_map,
    spatial_block_folds,
    write_sigma_geotiff,
)

try:  # optional, only needed for the GAM-vs-binned comparison test
    import pygam  # noqa: F401
    HAS_PYGAM = True
except ImportError:  # pragma: no cover
    HAS_PYGAM = False

rasterio = pytest.importorskip("rasterio")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _sigma_true(slope, roughness, dom_class=None):
    """The planted σ law: exponential in slope, linear in roughness, ×2 on buildings."""
    s = 0.05 * np.exp(0.03 * np.asarray(slope)) * (1.0 + 0.15 * np.asarray(roughness))
    if dom_class is not None:
        s = s * np.where(np.asarray(dom_class) == "building", 2.0, 1.0)
    return s


@pytest.fixture(scope="module")
def synth_frame():
    """60k pixels of stable terrain with a known heteroscedastic σ."""
    rng = np.random.default_rng(20260731)
    n = 60_000
    slope = rng.uniform(0.0, 60.0, n)
    roughness = rng.gamma(2.0, 1.0, n)
    dom_class = rng.choice(["ground", "building", "other"], n, p=[0.7, 0.15, 0.15])
    sig = _sigma_true(slope, roughness, dom_class)
    dh = rng.normal(0.0, 1.0, n) * sig + 0.02  # small datum offset, as in real data
    return pd.DataFrame({
        "dh": dh,
        "slope": slope,
        "roughness": roughness,
        "dom_class": pd.Categorical(dom_class),
        "x": rng.uniform(0.0, 3_000.0, n),
        "y": rng.uniform(0.0, 3_000.0, n),
        "sigma_true": sig,
    })


@pytest.fixture(scope="module")
def synth_raster():
    """A small gridded scene: dh, slope, roughness, stable mask, transform."""
    from affine import Affine

    rng = np.random.default_rng(7)
    ny, nx = 160, 200
    yy, xx = np.mgrid[0:ny, 0:nx]
    slope = 5.0 + 50.0 * (xx / nx)                       # slope ramps west->east
    roughness = 0.5 + 3.0 * np.abs(np.sin(yy / 12.0))    # roughness bands
    sig = _sigma_true(slope, roughness)
    dh = rng.normal(0.0, 1.0, (ny, nx)) * sig
    stable = np.ones((ny, nx), dtype=float)
    stable[:20, :20] = 0.0                               # a small unstable corner
    transform = Affine(1.0, 0.0, 500_000.0, 0.0, -1.0, 4_000_000.0)
    return dict(dh=dh, slope=slope, roughness=roughness, sigma_true=sig,
                stable=stable, transform=transform, shape=(ny, nx))


# ---------------------------------------------------------------------------
# nd_binning
# ---------------------------------------------------------------------------

class TestNdBinning:

    def test_returns_table_and_edges_with_expected_columns(self, synth_frame):
        table, edges = nd_binning(synth_frame, ["slope", "roughness"], n_bins=6)
        assert set(edges) == {"slope", "roughness"}
        for col in ("slope", "roughness", "slope_bin", "roughness_bin",
                    "count", "statistic"):
            assert col in table.columns
        # a 6x6 quantile grid on 60k well-spread rows should be fully occupied
        assert len(table) == 36
        assert table["count"].sum() == pytest.approx(len(synth_frame), rel=0.01)

    def test_bin_centres_lie_inside_their_edges(self, synth_frame):
        table, edges = nd_binning(synth_frame, ["slope"], n_bins=8)
        e = edges["slope"]
        k = table["slope_bin"].to_numpy()
        assert np.all(table["slope"].to_numpy() > e[k])
        assert np.all(table["slope"].to_numpy() < e[k + 1])

    def test_statistic_recovers_planted_sigma_marginally(self, synth_frame):
        """NMAD per slope bin should track the planted σ law."""
        table, _ = nd_binning(synth_frame, ["slope"], n_bins=8, min_count=100)
        expected = [
            mixture_nmad(synth_frame.loc[
                (synth_frame.slope > c - 4) & (synth_frame.slope < c + 4), "sigma_true"
            ])
            for c in table["slope"]
        ]
        assert np.allclose(table["statistic"], expected, rtol=0.12)

    def test_min_count_marks_thin_cells_nan_but_keeps_them(self, synth_frame):
        table, _ = nd_binning(synth_frame, ["slope", "roughness"],
                              n_bins=8, min_count=10_000)
        assert table["statistic"].isna().all()      # no 8x8 cell has 10k rows
        assert (table["count"] > 0).all()           # occupancy is still reported

    def test_outlier_guard_reduces_the_statistic(self, synth_frame):
        # contaminate heavily enough that even the robust NMAD statistic is
        # visibly inflated without the guard (30% of the smallest slope bin),
        # so the comparison is strict and a no-op guard fails the test
        spiked = synth_frame.copy()
        low = spiked.sort_values("slope").index[: int(0.3 * len(spiked) / 4)]
        spiked.loc[low, "dh"] = 500.0
        with_guard, _ = nd_binning(spiked, ["slope"], n_bins=4,
                                   fac_spread_outliers=7.0)
        without, _ = nd_binning(spiked, ["slope"], n_bins=4,
                                fac_spread_outliers=None)
        assert with_guard["statistic"].max() < without["statistic"].max()

    def test_supplied_edges_are_reused(self, synth_frame):
        _, edges = nd_binning(synth_frame, ["slope"], n_bins=5)
        held = synth_frame.iloc[:5_000]
        table2, edges2 = nd_binning(held, ["slope"], n_bins=99, edges=edges)
        assert np.array_equal(edges2["slope"], edges["slope"])
        assert table2["slope_bin"].max() <= len(edges["slope"]) - 2

    def test_rejects_unknown_columns(self, synth_frame):
        with pytest.raises(KeyError):
            nd_binning(synth_frame, ["not_a_column"])
        with pytest.raises(KeyError):
            nd_binning(synth_frame, ["slope"], value_col="nope")
        with pytest.raises(ValueError):
            nd_binning(synth_frame, [])

    def test_constant_predictor_collapses_to_one_bin(self, synth_frame):
        df = synth_frame.copy()
        df["flat"] = 3.0
        table, edges = nd_binning(df, ["flat"], n_bins=8, min_count=10)
        assert len(edges["flat"]) == 2
        assert len(table) == 1


# ---------------------------------------------------------------------------
# mixture_nmad
# ---------------------------------------------------------------------------

class TestMixtureNmad:

    def test_reduces_to_sigma_for_a_constant_scale(self):
        assert mixture_nmad(np.full(1000, 0.37)) == pytest.approx(0.37)

    def test_matches_a_simulated_mixture(self):
        rng = np.random.default_rng(3)
        sig = np.concatenate([np.full(50_000, 0.1), np.full(50_000, 1.0)])
        sample = rng.normal(0.0, 1.0, sig.size) * sig
        assert mixture_nmad(sig) == pytest.approx(nmad(sample), rel=0.03)

    def test_differs_from_the_median_sigma_for_a_heterogeneous_bin(self):
        """The whole point of the function: NMAD(mixture) != median(σ).

        A 50/50 mix of a narrow and a wide Gaussian has *half* its mass packed
        near zero, so ``median(|X|)`` (and hence the NMAD) sits well below the
        median σ.  Using ``median(σ̂)`` as the predicted dispersion would make
        such a bin look badly under-predicted when the model is in fact exact.
        """
        rng = np.random.default_rng(17)
        sig = np.concatenate([np.full(50_000, 0.05), np.full(50_000, 0.5)])
        simulated = nmad(rng.normal(0.0, 1.0, sig.size) * sig)
        assert mixture_nmad(sig) == pytest.approx(simulated, rel=0.05)
        assert abs(mixture_nmad(sig) / np.median(sig) - 1.0) > 0.10

    def test_empty_input_is_nan(self):
        assert math.isnan(mixture_nmad(np.array([])))
        assert math.isnan(mixture_nmad(np.array([np.nan, -1.0, 0.0])))


# ---------------------------------------------------------------------------
# BinnedSigmaModel / fit_binned_sigma_model
# ---------------------------------------------------------------------------

class TestFitBinnedSigmaModel:

    def test_recovers_the_planted_sigma_law(self, synth_frame):
        model = fit_binned_sigma_model(synth_frame, ["slope", "roughness"],
                                       ["dom_class"], n_bins=8, min_count=100)
        pred = model.predict(synth_frame)
        ratio = pred / synth_frame["sigma_true"].to_numpy()
        assert np.nanmedian(ratio) == pytest.approx(1.0, abs=0.05)
        # and it is right across the range, not only on average
        assert np.nanpercentile(ratio, 5) > 0.85
        assert np.nanpercentile(ratio, 95) < 1.15

    def test_grid_is_fully_observed_for_a_dense_two_predictor_fit(self, synth_frame):
        model = fit_binned_sigma_model(synth_frame, ["slope", "roughness"],
                                       n_bins=8, min_count=100)
        assert model.n_observed == 64
        assert model.n_filled == 0
        assert model.grid_values.shape == (8, 8)

    def test_sparse_grid_is_filled_and_warns(self, synth_frame):
        small = synth_frame.iloc[:3_000]
        with pytest.warns(UserWarning, match="had to be interpolated"):
            model = fit_binned_sigma_model(small, ["slope", "roughness"],
                                           n_bins=12, min_count=25)
        assert model.n_filled > 0
        assert model.n_observed > 0
        assert np.isfinite(model.grid_values).all()   # fill leaves no holes
        assert np.isfinite(model.predict(small)).all()

    def test_completely_unoccupied_grid_raises(self, synth_frame):
        with pytest.raises(ValueError, match="No bin reached min_count"):
            fit_binned_sigma_model(synth_frame.iloc[:3_000],
                                   ["slope", "roughness"],
                                   n_bins=12, min_count=1_000)

    def test_categorical_offset_recovers_the_planted_factor(self, synth_frame):
        model = fit_binned_sigma_model(synth_frame, ["slope", "roughness"],
                                       ["dom_class"], n_bins=8)
        off = model.factor_offsets["dom_class"]
        # planted building/ground ratio is 2.0 -> log offset difference ~ log 2,
        # attenuated by the n/(n+100) empirical-Bayes shrinkage
        assert np.exp(off["building"] - off["ground"]) == pytest.approx(2.0, rel=0.12)

    def test_predict_is_nan_where_a_predictor_is_nan(self, synth_frame):
        model = fit_binned_sigma_model(synth_frame, ["slope", "roughness"], n_bins=6)
        probe = synth_frame.head(10).copy()
        probe.loc[probe.index[:3], "roughness"] = np.nan
        pred = model.predict(probe)
        assert np.isnan(pred[:3]).all()
        assert np.isfinite(pred[3:]).all()

    def test_prediction_is_clipped_to_the_training_range(self, synth_frame):
        """Out-of-range predictors must saturate, not extrapolate to nonsense."""
        model = fit_binned_sigma_model(synth_frame, ["slope", "roughness"], n_bins=8)
        lo, hi = model.predictor_ranges["slope"]
        edge = pd.DataFrame({"slope": [hi], "roughness": [2.0]})
        far = pd.DataFrame({"slope": [hi * 1e3], "roughness": [2.0]})
        assert model.predict(far)[0] == pytest.approx(model.predict(edge)[0])
        assert np.isfinite(model.predict(far)[0])

    def test_predict_raises_on_missing_predictor_column(self, synth_frame):
        model = fit_binned_sigma_model(synth_frame, ["slope", "roughness"], n_bins=6)
        with pytest.raises(KeyError, match="Predictors missing"):
            model.predict(synth_frame[["dh", "slope"]])

    def test_linear_space_matches_log_space_closely(self, synth_frame):
        """log_space=False reproduces xDEM's raw-statistic interpolation."""
        a = fit_binned_sigma_model(synth_frame, ["slope", "roughness"],
                                   n_bins=8, log_space=True)
        b = fit_binned_sigma_model(synth_frame, ["slope", "roughness"],
                                   n_bins=8, log_space=False)
        pa, pb = a.predict(synth_frame), b.predict(synth_frame)
        assert np.nanmedian(pb / pa) == pytest.approx(1.0, abs=0.05)
        assert (pb > 0).all()          # raw-space fill must stay positive

    def test_all_predictors_unusable_raises(self, synth_frame):
        df = synth_frame.copy()
        df["blank"] = np.nan
        with pytest.raises(ValueError, match="No usable continuous predictors"):
            fit_binned_sigma_model(df, ["blank"])

    def test_min_count_too_high_raises_informatively(self, synth_frame):
        with pytest.raises(ValueError, match="min_count"):
            fit_binned_sigma_model(synth_frame, ["slope"], n_bins=8,
                                   min_count=10 ** 9)

    def test_is_duck_type_compatible_with_standardize(self, synth_frame):
        model = fit_binned_sigma_model(synth_frame, ["slope", "roughness"],
                                       ["dom_class"], n_bins=8)
        std = standardize(synth_frame, model)
        assert {"sigma_hat", "z"} <= set(std.columns)
        assert std.attrs["sigma_scale"] == pytest.approx(1.0, abs=0.1)
        stats = qq_stats(std["z"].to_numpy())
        assert stats["nmad"] == pytest.approx(1.0, abs=1e-6)   # by construction
        assert abs(stats["kurtosis_excess"]) < 0.5             # not by construction

    def test_is_duck_type_compatible_with_evaluate_sigma_raster(self, synth_raster):
        r = synth_raster
        frame = build_predictor_frame(
            r["dh"], r["slope"], np.zeros_like(r["slope"]), r["roughness"],
            pc_predictors={}, stable_mask=r["stable"], add_pixel_index=True,
        )
        model = fit_binned_sigma_model(frame, ["slope", "roughness"],
                                       n_bins=6, min_count=50)
        full = build_predictor_frame(
            r["dh"], r["slope"], np.zeros_like(r["slope"]), r["roughness"],
            pc_predictors={}, stable_mask=None, add_pixel_index=True,
        )
        sigma = evaluate_sigma_raster(model, full, r["shape"])
        assert sigma.shape == r["shape"]
        finite = np.isfinite(sigma)
        assert finite.mean() > 0.95
        ratio = sigma[finite] / r["sigma_true"][finite]
        assert np.median(ratio) == pytest.approx(1.0, abs=0.12)

    def test_sigma_increases_with_slope_as_planted(self, synth_frame):
        model = fit_binned_sigma_model(synth_frame, ["slope", "roughness"], n_bins=8)
        probe = pd.DataFrame({"slope": [5.0, 25.0, 50.0], "roughness": [2.0] * 3})
        pred = model.predict(probe)
        assert np.all(np.diff(pred) > 0)


# ---------------------------------------------------------------------------
# write_sigma_geotiff
# ---------------------------------------------------------------------------

class TestWriteSigmaGeotiff:

    def test_round_trip_preserves_grid_and_values(self, synth_raster, tmp_path):
        r = synth_raster
        sigma = r["sigma_true"].copy()
        sigma[0, 0] = np.nan
        path = str(tmp_path / "sigma.tif")
        out = write_sigma_geotiff(sigma, path, r["transform"], "EPSG:32610",
                                  tags={"predictors": "slope,roughness"})
        assert out == path
        with rasterio.open(path) as src:
            assert src.crs.to_string() == "EPSG:32610"
            assert src.transform == r["transform"]
            assert src.count == 1
            assert src.dtypes[0] == "float32"
            assert src.nodata == -9999.0
            assert src.tags()["predictors"] == "slope,roughness"
            assert src.tags()["UNITS"] == "m"
            assert src.descriptions[0].startswith("per-pixel")
            data = src.read(1, masked=True)
        assert data.mask[0, 0]                                   # NaN -> nodata
        assert np.allclose(data[1:], sigma[1:], rtol=1e-5)

    def test_rejects_non_2d(self, synth_raster, tmp_path):
        with pytest.raises(ValueError, match="2-D"):
            write_sigma_geotiff(np.zeros((2, 3, 4)), str(tmp_path / "x.tif"),
                                synth_raster["transform"], "EPSG:32610")

    def test_custom_nodata_is_honoured(self, synth_raster, tmp_path):
        sigma = np.full(synth_raster["shape"], np.nan)
        path = str(tmp_path / "allnodata.tif")
        write_sigma_geotiff(sigma, path, synth_raster["transform"], "EPSG:32610",
                            nodata=-1.0)
        with rasterio.open(path) as src:
            assert src.nodata == -1.0
            assert src.read(1, masked=True).mask.all()


# ---------------------------------------------------------------------------
# calibration_by_bin
# ---------------------------------------------------------------------------

class TestCalibrationByBin:

    def test_well_specified_model_calibrates_to_one(self, synth_frame):
        model = fit_binned_sigma_model(synth_frame, ["slope", "roughness"],
                                       ["dom_class"], n_bins=8)
        std = standardize(synth_frame, model)
        for predictor in ("slope", "roughness"):
            cal = calibration_by_bin(std, predictor, n_bins=8)
            assert len(cal) == 8
            assert np.allclose(cal["ratio"], 1.0, atol=0.10)
            assert np.allclose(cal["nmad_z"], 1.0, atol=0.10)
            assert np.allclose(cal["coverage_95"], 0.95, atol=0.03)

    def test_categorical_predictor_is_grouped_by_level(self, synth_frame):
        model = fit_binned_sigma_model(synth_frame, ["slope", "roughness"],
                                       ["dom_class"], n_bins=8)
        std = standardize(synth_frame, model)
        cal = calibration_by_bin(std, "dom_class")
        assert set(cal["bin"]) == {"ground", "building", "other"}
        assert np.allclose(cal["ratio"], 1.0, atol=0.10)

    def test_omitting_a_real_driver_shows_up_as_a_ratio_gradient(self, synth_frame):
        """A σ model blind to roughness must mis-calibrate across roughness bins."""
        model = fit_binned_sigma_model(synth_frame, ["slope"], ["dom_class"],
                                       n_bins=8)
        std = standardize(synth_frame, model)
        cal = calibration_by_bin(std, "roughness", n_bins=8)
        assert cal["ratio"].iloc[0] < 0.9        # low roughness: σ over-predicted
        assert cal["ratio"].iloc[-1] > 1.1       # high roughness: under-predicted
        assert cal["ratio"].is_monotonic_increasing

    def test_thin_bins_are_dropped(self, synth_frame):
        model = fit_binned_sigma_model(synth_frame, ["slope"], n_bins=6)
        std = standardize(synth_frame, model)
        assert calibration_by_bin(std, "slope", n_bins=6,
                                  min_count=10 ** 9).empty

    def test_unknown_predictor_raises(self, synth_frame):
        model = fit_binned_sigma_model(synth_frame, ["slope"], n_bins=6)
        std = standardize(synth_frame, model)
        with pytest.raises(KeyError):
            calibration_by_bin(std, "nope")


# ---------------------------------------------------------------------------
# spatial_block_folds
# ---------------------------------------------------------------------------

class TestSpatialBlockFolds:

    def test_partitions_all_rows_into_the_requested_folds(self, synth_frame):
        folds = spatial_block_folds(synth_frame["x"], synth_frame["y"],
                                    n_folds=5, block_size=400.0, seed=0)
        assert folds.shape == (len(synth_frame),)
        assert set(np.unique(folds)) == set(range(5))

    def test_is_deterministic_for_a_fixed_seed(self, synth_frame):
        a = spatial_block_folds(synth_frame["x"], synth_frame["y"], seed=11)
        b = spatial_block_folds(synth_frame["x"], synth_frame["y"], seed=11)
        c = spatial_block_folds(synth_frame["x"], synth_frame["y"], seed=12)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_points_in_the_same_block_share_a_fold(self):
        x = np.array([10.0, 20.0, 30.0, 610.0, 620.0])
        y = np.zeros(5)
        folds = spatial_block_folds(x, y, n_folds=2, block_size=500.0, seed=0)
        assert folds[0] == folds[1] == folds[2]
        assert folds[3] == folds[4]

    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            spatial_block_folds(np.arange(5.0), np.arange(4.0))
        with pytest.raises(ValueError):
            spatial_block_folds(np.arange(5.0), np.arange(5.0), block_size=0.0)


# ---------------------------------------------------------------------------
# cross_validate_sigma_models
# ---------------------------------------------------------------------------

class TestCrossValidateSigmaModels:

    def test_scores_a_correct_model_near_its_targets(self, synth_frame):
        cv = cross_validate_sigma_models(
            synth_frame,
            {"binned": lambda d: fit_binned_sigma_model(
                d, ["slope", "roughness"], ["dom_class"], n_bins=6)},
            predictors=["slope", "roughness"], n_folds=4, block_size=800.0,
        )
        assert len(cv) == 4
        assert (cv["error"] == "").all()
        assert np.allclose(cv["nmad_z"], 1.0, atol=0.10)
        assert np.allclose(cv["coverage_95"], 0.95, atol=0.03)
        assert (cv["calibration_rmse"] < 0.12).all()
        assert (cv["frac_predicted"] > 0.99).all()

    def test_prefers_the_model_that_sees_the_real_driver(self, synth_frame):
        """A σ model missing 'roughness' must score worse on every criterion."""
        cv = cross_validate_sigma_models(
            synth_frame,
            {
                "full": lambda d: fit_binned_sigma_model(
                    d, ["slope", "roughness"], ["dom_class"], n_bins=6),
                "slope_only": lambda d: fit_binned_sigma_model(
                    d, ["slope"], ["dom_class"], n_bins=6),
            },
            predictors=["slope", "roughness"], n_folds=4, block_size=800.0,
        )
        agg = cv.groupby("model")[["nll", "calibration_rmse"]].mean()
        assert agg.loc["full", "nll"] < agg.loc["slope_only", "nll"]
        assert agg.loc["full", "calibration_rmse"] < agg.loc["slope_only", "calibration_rmse"]

    def test_a_failing_fitter_is_recorded_not_raised(self, synth_frame):
        def broken(_d):
            raise RuntimeError("deliberate")

        cv = cross_validate_sigma_models(
            synth_frame,
            {"ok": lambda d: fit_binned_sigma_model(d, ["slope"], n_bins=6),
             "broken": broken},
            predictors=["slope"], n_folds=3, block_size=1_000.0,
        )
        broke = cv[cv["model"] == "broken"]
        assert len(broke) == 3
        assert broke["error"].str.contains("deliberate").all()
        assert broke["nmad_z"].isna().all()
        assert (cv[cv["model"] == "ok"]["error"] == "").all()

    def test_requires_coordinates(self, synth_frame):
        with pytest.raises(KeyError, match="'x'"):
            cross_validate_sigma_models(
                synth_frame.drop(columns=["x"]),
                {"m": lambda d: fit_binned_sigma_model(d, ["slope"])},
            )

    @pytest.mark.skipif(not HAS_PYGAM, reason="pygam not installed")
    def test_gam_and_binned_are_both_scorable_head_to_head(self, synth_frame):
        from topochange.heteroscedastic import fit_sigma_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = cross_validate_sigma_models(
                synth_frame,
                {
                    "binned": lambda d: fit_binned_sigma_model(
                        d, ["slope", "roughness"], ["dom_class"], n_bins=6),
                    "gam": lambda d: fit_sigma_model(
                        d, predictors=["slope", "roughness"],
                        factor_cols=["dom_class"], n_bins=6),
                },
                predictors=["slope", "roughness"], n_folds=3, block_size=1_000.0,
            )
        agg = cv.groupby("model")[["nmad_z", "nll", "calibration_rmse"]].mean()
        assert set(agg.index) == {"binned", "gam"}
        assert agg["nll"].notna().all()
        # both estimators see the same planted law, so both should calibrate
        assert (agg["nmad_z"] - 1.0).abs().max() < 0.15


# ---------------------------------------------------------------------------
# patch_validation
# ---------------------------------------------------------------------------

class TestPatchValidation:

    def test_uncorrelated_field_follows_the_root_n_law(self, synth_raster):
        """With no spatial correlation, σ(mean) must fall as 1/√n."""
        r = synth_raster
        res = patch_validation(r["dh"], r["transform"], stable_mask=r["stable"],
                               sigma=r["sigma_true"],
                               areas=[25.0, 100.0, 400.0], n_patches=400, seed=1)
        assert list(res["patch_size_px"]) == [5, 10, 20]
        assert res["n_patches"].min() > 100
        # empirical dispersion of patch means matches the independent prediction
        assert np.allclose(res["correlation_inflation"], 1.0, atol=0.20)
        # and halves each time the patch area quadruples
        ratios = res["empirical_sigma_mean"].to_numpy()
        assert ratios[0] / ratios[1] == pytest.approx(2.0, rel=0.25)
        assert ratios[1] / ratios[2] == pytest.approx(2.0, rel=0.25)

    def test_correlated_field_inflates_above_the_independent_prediction(self):
        """A smoothed (correlated) field must exceed the 1/√n expectation."""
        from affine import Affine
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(5)
        ny, nx = 200, 200
        white = rng.normal(0.0, 1.0, (ny, nx))
        smooth = gaussian_filter(white, sigma=6.0, mode="wrap")
        smooth = smooth / np.std(smooth)              # unit per-pixel variance
        sigma = np.ones((ny, nx))
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        res = patch_validation(smooth, transform, sigma=sigma,
                               areas=[400.0], n_patches=500, seed=2)
        assert res["correlation_inflation"].iloc[0] > 3.0

    def test_stable_mask_excludes_the_unstable_corner(self, synth_raster):
        r = synth_raster
        spiked = r["dh"].copy()
        spiked[:20, :20] = 50.0                       # huge "real change"
        masked = patch_validation(spiked, r["transform"], stable_mask=r["stable"],
                                  areas=[100.0], n_patches=300, seed=3)
        unmasked = patch_validation(spiked, r["transform"],
                                    areas=[100.0], n_patches=300, seed=3)
        assert masked["empirical_sigma_mean"].iloc[0] < \
            unmasked["empirical_sigma_mean"].iloc[0]

    def test_patch_larger_than_the_raster_returns_an_empty_row(self, synth_raster):
        r = synth_raster
        res = patch_validation(r["dh"], r["transform"], areas=[1e9], n_patches=10)
        assert res["n_patches"].iloc[0] == 0
        assert np.isnan(res["empirical_sigma_mean"].iloc[0])

    def test_validates_shapes(self, synth_raster):
        r = synth_raster
        with pytest.raises(ValueError, match="2-D"):
            patch_validation(np.zeros((2, 2, 2)), r["transform"])
        with pytest.raises(ValueError, match="stable_mask"):
            patch_validation(r["dh"], r["transform"], stable_mask=np.ones((3, 3)))
        with pytest.raises(ValueError, match="sigma"):
            patch_validation(r["dh"], r["transform"], sigma=np.ones((3, 3)))


# ---------------------------------------------------------------------------
# plotting (smoke tests; they must not raise)
# ---------------------------------------------------------------------------

class TestPlots:

    def test_plot_helpers_run(self, synth_frame, synth_raster):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        table, _ = nd_binning(synth_frame, ["slope", "roughness"], n_bins=6)
        assert plot_1d_binning(table, "slope", label="stable") is not None
        assert plot_2d_binning(table, "slope", "roughness") is not None
        assert plot_sigma_map(synth_raster["sigma_true"],
                              transform=synth_raster["transform"]) is not None
        plt.close("all")

    def test_plot_sigma_map_rejects_an_all_nan_field(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        with pytest.raises(ValueError, match="non-finite"):
            plot_sigma_map(np.full((4, 4), np.nan))


# ---------------------------------------------------------------------------
# interoperability helpers (xDEM cross-check)
# ---------------------------------------------------------------------------

class TestCallableSigmaModel:

    def test_wraps_a_plain_function(self):
        m = CallableSigmaModel(lambda v: 0.05 + 0.002 * np.asarray(v[0]), ["slope"])
        out = m.predict(pd.DataFrame({"slope": [0.0, 10.0, 50.0]}))
        assert np.allclose(out, [0.05, 0.07, 0.15])

    def test_argument_order_follows_predictors(self):
        m = CallableSigmaModel(lambda v: np.asarray(v[0]) - np.asarray(v[1]), ["a", "b"])
        df = pd.DataFrame({"a": [10.0], "b": [3.0]})
        assert m.predict(df)[0] == pytest.approx(7.0)
        m2 = CallableSigmaModel(lambda v: np.asarray(v[0]) - np.asarray(v[1]), ["b", "a"])
        assert m2.predict(df)[0] == pytest.approx(-7.0)

    def test_nan_predictors_are_skipped_not_passed_through(self):
        seen = {}

        def f(v):
            seen["n"] = len(v[0])
            return np.asarray(v[0])

        m = CallableSigmaModel(f, ["slope"])
        out = m.predict(pd.DataFrame({"slope": [1.0, np.nan, 3.0]}))
        assert seen["n"] == 2                    # the NaN row never reached fun
        assert np.isnan(out[1])
        assert np.allclose(out[[0, 2]], [1.0, 3.0])

    def test_missing_column_raises(self):
        m = CallableSigmaModel(lambda v: np.asarray(v[0]), ["slope"])
        with pytest.raises(KeyError, match="Predictors missing"):
            m.predict(pd.DataFrame({"other": [1.0]}))

    def test_is_usable_by_standardize_and_cross_validation(self, synth_frame):
        """The whole point: a foreign estimator must score on the same footing."""
        # The oracle sigma law, with the categorical effect supplied through the
        # adapter's factor mechanism (exercising that path too).
        m = CallableSigmaModel(
            lambda v: _sigma_true(v[0], v[1]), ["slope", "roughness"],
            factor_cols=["dom_class"],
            factor_offsets={"dom_class": {"building": float(np.log(2.0))}},
        )
        assert m.predict(synth_frame)[:200] == pytest.approx(
            synth_frame["sigma_true"].to_numpy()[:200], rel=1e-9)
        std = standardize(synth_frame, m)
        assert std.attrs["sigma_scale"] == pytest.approx(1.0, abs=0.05)
        cv = cross_validate_sigma_models(
            synth_frame, {"oracle": lambda d: m},
            predictors=["slope", "roughness"], n_folds=3, block_size=1_000.0)
        assert (cv["error"] == "").all()
        assert np.allclose(cv["nmad_z"], 1.0, atol=0.10)

    def test_all_nan_input_returns_all_nan(self):
        m = CallableSigmaModel(lambda v: np.asarray(v[0]), ["slope"])
        assert np.isnan(m.predict(pd.DataFrame({"slope": [np.nan, np.nan]}))).all()


class TestUniformEdges:

    def test_edges_are_equally_spaced_and_span_the_range(self, synth_frame):
        e = uniform_edges(synth_frame, ["slope"], n_bins=10)["slope"]
        assert len(e) == 11
        assert np.allclose(np.diff(e), np.diff(e)[0], rtol=1e-6)
        assert e[0] <= synth_frame["slope"].min()
        assert e[-1] >= synth_frame["slope"].max()

    def test_differs_from_the_default_quantile_edges_on_skewed_data(self, synth_frame):
        """The reason the default is quantile: uniform edges unbalance the counts."""
        uni = uniform_edges(synth_frame, ["roughness"], n_bins=10)
        tab_u, _ = nd_binning(synth_frame, ["roughness"], n_bins=10,
                              min_count=1, edges=uni)
        tab_q, _ = nd_binning(synth_frame, ["roughness"], n_bins=10, min_count=1)
        spread = lambda t: t["count"].max() / max(t["count"].min(), 1)
        assert spread(tab_u) > 10 * spread(tab_q)

    def test_feeds_fit_binned_sigma_model(self, synth_frame):
        e = uniform_edges(synth_frame, ["slope", "roughness"], n_bins=10)
        m = fit_binned_sigma_model(synth_frame, ["slope", "roughness"],
                                   n_bins=10, edges=e, log_space=False)
        assert np.allclose(m.grid_points[0], 0.5 * (e["slope"][:-1] + e["slope"][1:]))

    def test_empty_predictor_raises(self, synth_frame):
        df = synth_frame.copy()
        df["blank"] = np.nan
        with pytest.raises(ValueError, match="No finite values"):
            uniform_edges(df, ["blank"])


class TestTrimmedNmadScale:

    def test_equals_plain_nmad_on_clean_data(self):
        z = np.random.default_rng(1).normal(0.0, 1.0, 50_000)
        assert trimmed_nmad_scale(z) == pytest.approx(nmad(z), rel=1e-3)

    def test_is_smaller_when_outliers_are_present(self):
        z = np.concatenate([np.random.default_rng(2).normal(0.0, 1.0, 20_000),
                            np.full(400, 500.0)])
        assert trimmed_nmad_scale(z) < nmad(z)

    def test_none_disables_the_trim(self):
        z = np.concatenate([np.random.default_rng(3).normal(0.0, 1.0, 5_000),
                            np.full(200, 500.0)])
        assert trimmed_nmad_scale(z, None) == pytest.approx(nmad(z))

    def test_empty_is_nan(self):
        assert math.isnan(trimmed_nmad_scale(np.array([np.nan, np.inf])))


# ---------------------------------------------------------------------------
# misregistration_diagnostic
# ---------------------------------------------------------------------------

class TestMisregistrationDiagnostic:

    @staticmethod
    def _scene(shift_m=0.0, direction_deg=0.0, slope_bias=0.0, seed=0, n=80_000,
               couple_aspect=False):
        """Stable terrain with a planted horizontal shift and/or slope-only bias.

        A horizontal shift of magnitude ``a`` toward azimuth ``b`` gives
        ``dh = a * cos(b - aspect) * tan(slope)`` (Nuth & Kaab 2011). The
        slope-only term is aspect-independent, so the two are separable in
        principle -- but only if the fit models both.

        ``couple_aspect`` ties aspect to slope, as in real valley terrain where
        steep ground faces a preferred direction. That coupling is what breaks
        the classical single-term fit, so most tests below run with it on.
        """
        rng = np.random.default_rng(seed)
        slope = rng.uniform(5.0, 45.0, n)
        if couple_aspect:
            centre = 20.0 + 180.0 * (slope - 5.0) / 40.0
            aspect = (centre + rng.normal(0.0, 25.0, n)) % 360.0
        else:
            aspect = rng.uniform(0.0, 360.0, n)
        a_r, b_r = np.radians(aspect), np.radians(direction_deg)
        dh = (shift_m * np.cos(b_r - a_r) * np.tan(np.radians(slope))
              + slope_bias * (slope / 45.0)
              + rng.normal(0.0, 0.05, n))
        return pd.DataFrame({"dh": dh, "slope": slope, "aspect": aspect})

    def test_recovers_a_planted_shift_with_uniform_aspect(self):
        r = misregistration_diagnostic(self._scene(shift_m=0.30, direction_deg=120.0))
        assert r["shift_m"] == pytest.approx(0.30, abs=0.02)
        assert r["direction_deg"] == pytest.approx(120.0, abs=3.0)
        assert r["frac_aspect"] > 0.95
        assert r["slope_bias_span"] < 0.02

    def test_recovers_a_planted_shift_despite_aspect_slope_coupling(self):
        """The case the classical single-term fit gets wrong."""
        r = misregistration_diagnostic(
            self._scene(shift_m=0.40, direction_deg=45.0, couple_aspect=True))
        assert r["aspect_slope_coupling"] > 0.5      # the hard regime
        assert r["shift_m"] == pytest.approx(0.40, abs=0.03)
        assert r["direction_deg"] == pytest.approx(45.0, abs=4.0)
        assert r["frac_aspect"] > 0.95

    def test_slope_only_bias_is_NOT_reported_as_a_shift(self):
        """The regression this fit exists to prevent.

        Without the slope nuisance term, coupled terrain with a purely
        slope-dependent bias produced a spurious ~0.6 m shift.
        """
        r = misregistration_diagnostic(
            self._scene(shift_m=0.0, slope_bias=0.40, couple_aspect=True))
        assert r["shift_m"] < 0.03
        assert r["frac_aspect"] < 0.05
        assert r["slope_bias_span"] == pytest.approx(0.40 * 40.0 / 45.0, abs=0.05)

    def test_separates_a_shift_from_a_coincident_slope_bias(self):
        r = misregistration_diagnostic(
            self._scene(shift_m=0.30, direction_deg=200.0, slope_bias=0.30,
                        couple_aspect=True))
        assert r["shift_m"] == pytest.approx(0.30, abs=0.03)
        assert r["direction_deg"] == pytest.approx(200.0, abs=4.0)
        assert r["slope_bias_span"] == pytest.approx(0.30 * 40.0 / 45.0, abs=0.06)
        assert 0.2 < r["frac_aspect"] < 0.95

    def test_clean_scene_reports_nothing(self):
        r = misregistration_diagnostic(self._scene())
        assert r["shift_m"] < 0.02
        assert r["slope_bias_span"] < 0.02
        assert abs(r["r2"]) < 0.05

    def test_slope_poly_zero_is_the_classical_form(self):
        """Documenting the failure mode: degree 0 reproduces the old behaviour."""
        df = self._scene(shift_m=0.0, slope_bias=0.40, couple_aspect=True)
        classical = misregistration_diagnostic(df, slope_poly=0)
        joint = misregistration_diagnostic(df, slope_poly=2)
        assert classical["shift_m"] > 5 * joint["shift_m"]

    def test_flat_terrain_is_excluded(self):
        df = self._scene(shift_m=0.20, direction_deg=90.0)
        flat = df.iloc[:5_000].copy()
        flat["slope"] = 0.01
        flat["dh"] = 0.3
        r = misregistration_diagnostic(pd.concat([df, flat], ignore_index=True))
        assert r["shift_m"] == pytest.approx(0.20, abs=0.03)
        assert r["n"] <= len(df)

    def test_requires_aspect(self):
        with pytest.raises(KeyError, match="aspect"):
            misregistration_diagnostic(self._scene()[["dh", "slope"]])

    def test_too_few_rows_returns_nan_not_an_exception(self):
        r = misregistration_diagnostic(self._scene(n=20))
        assert math.isnan(r["shift_m"])
        assert r["n"] < 100


class TestBinnedBiasModel:
    def test_recovers_planted_predictor_dependent_bias(self):
        from topochange.sigma_map import fit_binned_bias_model
        rng = np.random.default_rng(7)
        n = 40_000
        slope = rng.uniform(0, 40, n)
        mu = -0.2 + 0.01 * slope           # planted terrain-dependent bias
        dh = mu + rng.normal(0, 0.05, n)
        df = pd.DataFrame({"dh": dh, "slope": slope})
        m = fit_binned_bias_model(df, ["slope"], n_bins=8, min_count=100)
        q = pd.DataFrame({"slope": [5.0, 20.0, 35.0]})
        pred = m.predict(q)
        assert np.allclose(pred, -0.2 + 0.01 * q["slope"].to_numpy(), atol=0.01)

    def test_missing_predictor_raises(self):
        from topochange.sigma_map import fit_binned_bias_model
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"dh": rng.normal(size=2000),
                           "slope": rng.uniform(0, 30, 2000)})
        m = fit_binned_bias_model(df, ["slope"], n_bins=4, min_count=50)
        with pytest.raises(KeyError):
            m.predict(pd.DataFrame({"roughness": [1.0]}))
