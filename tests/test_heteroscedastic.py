"""Tests for the heteroscedastic (spatially-varying σ) module.

Coverage
--------
1. Robust statistics (NMAD) and flight-strip detection.
2. ClassificationSpec filter rules / labels.
3. AnisotropicCompositeVariogram: isotropic reduction, anisotropy convention,
   correlogram limits.
4. Directional empirical variogram (omnidirectional + sectored).
5. Heteroscedastic areal uncertainty (MC vs pairwise consistency; scaling
   with correlation range).
6. build_predictor_frame masking / pixel indexing.
7. fit_anisotropic_variogram range recovery (scipy).
8. fit_sigma_model + σ prediction (pygam-gated).
9. extract_pointcloud_predictors on a synthetic LAZ (PDAL-gated).

scipy and shapely are core project dependencies, so most tests run
unconditionally; ``pygam`` and ``pdal`` tests are skipped when those optional
dependencies are absent.
"""

import json

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from topochange.composite_variogram import CompositeVariogramModel
from topochange.heteroscedastic import (
    AnisotropicCompositeVariogram,
    ClassificationSpec,
    HeteroscedasticUncertaintyEstimator,
    build_predictor_frame,
    derive_strip_ids,
    derive_strip_ids_from_point_source_id,
    directional_empirical_variogram,
    fit_anisotropic_variogram,
    fit_sigma_model,
    is_point_source_id_informative,
    nmad,
    standardize,
)
from skip_markers import requires_pdal, SYNTHETIC


def _has_pygam() -> bool:
    try:
        import pygam  # noqa: F401
        return True
    except ImportError:
        return False


requires_pygam = pytest.mark.skipif(not _has_pygam(), reason="pygam is not installed")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _Affine:
    """Minimal north-up affine (1 m pixels) sufficient for the areal tests.

    Avoids a hard dependency on ``affine`` in the test itself; the real code
    receives a rasterio ``Affine`` which supports the same operations.
    """

    def __init__(self, a, b, c, d, e, f):
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f

    def __mul__(self, xy):
        x, y = xy
        return (self.a * x + self.b * y + self.c, self.d * x + self.e * y + self.f)

    def __invert__(self):
        return _Affine(1.0 / self.a, 0.0, -self.c / self.a,
                       0.0, 1.0 / self.e, -self.f / self.e)


def _spherical_composite(sill=1.0, rng_=100.0, nug=0.0):
    """A fitted single-component spherical (+nugget) composite."""
    comp = CompositeVariogramModel(["spherical"], include_nugget=True)
    comp.set_params([sill, rng_, nug])
    return comp


def _item(v):
    return float(np.asarray(v).ravel()[0])


# ---------------------------------------------------------------------------
# robust stats + strip ids
# ---------------------------------------------------------------------------

def test_nmad_matches_std_on_normal():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 2.0, 200_000)
    assert abs(nmad(x) - 2.0) < 0.03


def test_nmad_ignores_nonfinite_and_empty():
    assert nmad([1.0, 2.0, np.nan, 3.0, np.inf]) > 0
    assert np.isnan(nmad([np.nan, np.inf]))
    assert np.isnan(nmad([]))


def test_derive_strip_ids_two_flightlines():
    t = np.concatenate([np.linspace(100, 101, 50), np.linspace(200, 201, 50)])
    ids = derive_strip_ids(t, gap_seconds=5.0)
    assert set(ids.tolist()) == {0, 1}
    assert (ids[:50] == 0).all() and (ids[50:] == 1).all()


def test_derive_strip_ids_marks_nan():
    ids = derive_strip_ids(np.array([np.nan, 1.0, 2.0]), gap_seconds=5.0)
    assert ids[0] == -1
    assert set(ids[1:].tolist()) == {0}


def test_derive_strip_ids_from_point_source_id_relabels_compactly():
    # raw LAS PointSourceId values need not be contiguous (e.g. flight-line
    # numbers assigned by the acquisition vendor); -1 marks "no coverage".
    raw = np.array([4540, 4540, 4542, 4541, -1, 4542])
    ids = derive_strip_ids_from_point_source_id(raw)
    assert ids[4] == -1
    assert ids[0] == ids[1]                        # same raw id -> same label
    assert len({ids[0], ids[2], ids[3]}) == 3       # three distinct raw ids -> three labels
    # relabelled in ascending order of the raw id: 4540->0, 4541->1, 4542->2
    assert (ids[0], ids[3], ids[2]) == (0, 1, 2)
    assert set(ids.tolist()) == {-1, 0, 1, 2}


def test_derive_strip_ids_from_point_source_id_all_missing():
    ids = derive_strip_ids_from_point_source_id(np.array([-1, -1, -1]))
    assert set(ids.tolist()) == {-1}


def test_is_point_source_id_informative():
    assert not is_point_source_id_informative(np.array([-1, -1, -1]))
    assert not is_point_source_id_informative(np.array([0, 0, 0, -1]))  # single constant id
    assert is_point_source_id_informative(np.array([0, 1, -1]))
    assert not is_point_source_id_informative(np.array([5, 5, -1]), min_strips=2)
    assert is_point_source_id_informative(np.array([5, 5, 6, -1]), min_strips=2)


# ---------------------------------------------------------------------------
# ClassificationSpec
# ---------------------------------------------------------------------------

def test_classification_filter_rules():
    assert ClassificationSpec(veg=True).pdal_filter_expression() == \
        "(ReturnNumber == NumberOfReturns)"
    assert ClassificationSpec(building=True).pdal_filter_expression() is None
    assert ClassificationSpec(ground=True).pdal_filter_expression() == \
        "(ReturnNumber == NumberOfReturns)"
    assert ClassificationSpec().pdal_filter_expression() is None
    # veg takes precedence over building/ground
    assert ClassificationSpec(veg=True, building=True).pdal_filter_expression() == \
        "(ReturnNumber == NumberOfReturns)"


def test_classification_labels():
    spec = ClassificationSpec(veg=True, building=True, ground=True)
    assert spec.label_for_point(2) == "ground"
    # vegetation is deliberately NOT a label category: canopy enters the sigma
    # model as a continuous predictor (e.g. a CHM via extra_rasters), so veg
    # class codes fall through to "other" (see ClassificationSpec.label_for_point)
    assert spec.label_for_point(5) == "other"
    assert spec.label_for_point(6) == "building"
    assert spec.label_for_point(1) == "other"


# ---------------------------------------------------------------------------
# AnisotropicCompositeVariogram
# ---------------------------------------------------------------------------

def test_anisotropic_reduces_to_isotropic():
    comp = _spherical_composite(sill=1.0, rng_=100.0)
    av = AnisotropicCompositeVariogram(composite=comp)  # ratio == 1
    h = np.array([5.0, 25.0, 75.0, 150.0])
    assert np.allclose(av.gamma(h), comp(h))
    assert abs(av.total_sill() - 1.0) < 1e-12


def test_anisotropy_longer_range_along_axis():
    comp = _spherical_composite(sill=1.0, rng_=100.0)
    av = AnisotropicCompositeVariogram(
        composite=comp, anisotropy_ratio=2.0, anisotropy_angle_deg=0.0
    )
    # ratio>1 => longer range along the axis (0 deg) => smaller gamma there
    g_along = _item(av.gamma(50.0, np.array([0.0])))
    g_across = _item(av.gamma(50.0, np.array([90.0])))
    assert g_along < g_across


def test_correlogram_limits():
    comp = _spherical_composite(sill=2.0, rng_=100.0)
    av = AnisotropicCompositeVariogram(composite=comp)
    assert abs(_item(av.correlogram(0.0)) - 1.0) < 1e-9  # ρ(0) = 1
    assert abs(_item(av.correlogram(1e6))) < 1e-9         # ρ(∞) = 0


def test_nonstationary_composite_rejected():
    comp = CompositeVariogramModel(["power"], include_nugget=False)
    comp.set_params([0.5, 1.2])
    av = AnisotropicCompositeVariogram(composite=comp)
    with pytest.raises(ValueError):
        av.total_sill()


# ---------------------------------------------------------------------------
# directional empirical variogram
# ---------------------------------------------------------------------------

def _synthetic_field(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 500, n)
    ys = rng.uniform(0, 500, n)
    z = np.sin(xs / 80.0) + np.cos(ys / 80.0) + rng.normal(0, 0.1, n)
    return pd.DataFrame({"x": xs, "y": ys, "z": z})


def test_directional_variogram_omnidirectional_increases():
    emp = directional_empirical_variogram(_synthetic_field(), n_pairs=150_000,
                                          n_bins=15, seed=1)
    assert (emp["direction"] == -1).all()
    assert emp["gamma"].iloc[0] < emp["gamma"].iloc[-1]


def test_directional_variogram_sectors():
    emp = directional_empirical_variogram(
        _synthetic_field(), directions=[(0, 22.5), (90, 22.5)],
        n_pairs=150_000, seed=1,
    )
    assert set(emp["direction"].unique()) == {0.0, 90.0}


def test_directional_variogram_requires_coords():
    with pytest.raises(ValueError):
        directional_empirical_variogram(pd.DataFrame({"z": [1, 2, 3]}))


# ---------------------------------------------------------------------------
# heteroscedastic areal uncertainty
# ---------------------------------------------------------------------------

def _constant_sigma_setup(sigma=1.0, rng_=100.0):
    transform = _Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)  # 1 m pixels
    sig = np.full((100, 100), sigma, dtype=float)
    vgm = AnisotropicCompositeVariogram(composite=_spherical_composite(1.0, rng_))
    est = HeteroscedasticUncertaintyEstimator(sig, transform, vgm)
    foi = Polygon([(0.5, 60.5), (39.5, 60.5), (39.5, 99.5), (0.5, 99.5)])
    return est, foi


def test_areal_mc_pairwise_consistent():
    est, foi = _constant_sigma_setup()
    mc = est.areal_uncertainty_mc(foi, n_realizations=400, seed=3)
    pw = est.areal_uncertainty_pairwise(foi, n_pairs=100_000, seed=3)
    n = mc["n_pixels"]
    # bounded below by the uncorrelated limit and above by the fully-correlated
    assert 1.0 / np.sqrt(n) < mc["sigma_area"] < 1.0
    assert abs(mc["sigma_area"] - pw["sigma_area"]) < 0.05


def test_areal_increases_with_range():
    short, foi = _constant_sigma_setup(rng_=100.0)
    long_, _ = _constant_sigma_setup(rng_=400.0)
    s = short.areal_uncertainty_mc(foi, n_realizations=400, seed=3)["sigma_area"]
    l = long_.areal_uncertainty_mc(foi, n_realizations=400, seed=3)["sigma_area"]
    assert l > s


def test_areal_too_few_pixels():
    est, _ = _constant_sigma_setup()
    tiny = Polygon([(0.5, 99.0), (2.5, 99.0), (2.5, 99.9), (0.5, 99.9)])
    out = est.areal_uncertainty_mc(tiny, n_realizations=50, seed=0)
    assert np.isnan(out["sigma_area"]) and "note" in out


# ---------------------------------------------------------------------------
# predictor frame
# ---------------------------------------------------------------------------

def test_build_predictor_frame_masks_and_indexes():
    dh = np.array([[0.1, 0.2], [np.nan, 0.4]])
    slope = np.full((2, 2), 10.0)
    aspect = np.zeros((2, 2))
    rough = np.ones((2, 2))
    pcp = {
        "density": np.array([[5, 6], [7, 8]], float),
        "dom_class": np.array([[1, 3], [2, 0]], np.int8),
    }
    fr = build_predictor_frame(dh, slope, aspect, rough, pcp, add_pixel_index=True)
    assert len(fr) == 3  # the NaN dh pixel is dropped
    assert "pixel_index" in fr.columns
    # "veg" is deliberately not a label; canopy enters as a continuous predictor
    assert set(fr["dom_class"].astype(str)) <= {"ground", "building", "other"}


def test_build_predictor_frame_stable_mask():
    dh = np.ones((2, 2))
    mask = np.array([[1, 0], [0, 1]])
    fr = build_predictor_frame(dh, np.zeros((2, 2)), np.zeros((2, 2)),
                               np.ones((2, 2)), {}, stable_mask=mask)
    assert len(fr) == 2


# ---------------------------------------------------------------------------
# anisotropic variogram fit (scipy)
# ---------------------------------------------------------------------------

def test_fit_anisotropic_variogram_recovers_range():
    # synthetic isotropic spherical empirical variogram, range=120, sill=1
    from topochange.variogram_models import spherical
    lags = np.linspace(2, 300, 40)
    g = spherical(lags, sill=1.0, range_=120.0)
    emp = pd.DataFrame({"lag": lags, "gamma": g, "n": np.full(lags.size, 500)})
    model = fit_anisotropic_variogram(emp, component_names=["spherical"],
                                      anisotropic=False)
    assert model.anisotropy_ratio == 1.0
    # total sill close to 1; practical range in a sensible band
    assert 0.7 < model.total_sill() < 1.4
    fitted_range = model.composite.get_component_params(0)[1]
    assert 60.0 < fitted_range < 200.0


# ---------------------------------------------------------------------------
# sigma model (pygam)
# ---------------------------------------------------------------------------

@requires_pygam
def test_fit_sigma_model_tracks_slope():
    # dh whose scale grows with slope; σ model should predict larger σ at high slope
    rng = np.random.default_rng(0)
    n = 20_000
    slope = rng.uniform(0, 40, n)
    roughness = rng.uniform(0, 2, n)
    sigma_true = 0.1 + 0.02 * slope
    dh = rng.normal(0, 1, n) * sigma_true
    df = pd.DataFrame({
        "dh": dh, "slope": slope, "roughness": roughness,
        "dom_class": pd.Categorical(["ground"] * n),
    })
    model = fit_sigma_model(df, predictors=["slope", "roughness"], factor_cols=[])
    lo = pd.DataFrame({"slope": [5.0], "roughness": [1.0]})
    hi = pd.DataFrame({"slope": [35.0], "roughness": [1.0]})
    s_lo = model.predict(lo)[0]
    s_hi = model.predict(hi)[0]
    assert s_lo > 0 and s_hi > 0
    assert s_hi > s_lo  # σ increases with slope


@requires_pygam
def test_standardize_yields_unit_scale():
    rng = np.random.default_rng(1)
    n = 20_000
    slope = rng.uniform(0, 40, n)
    sigma_true = 0.2 + 0.03 * slope
    dh = rng.normal(0, 1, n) * sigma_true
    df = pd.DataFrame({"dh": dh, "slope": slope, "roughness": rng.uniform(0, 1, n),
                       "dom_class": pd.Categorical(["ground"] * n)})
    model = fit_sigma_model(df, predictors=["slope"], factor_cols=[])
    # rescale=False: the default two-step rescale forces NMAD(z)==1 by
    # construction, which would make this test unfalsifiable. Without it,
    # NMAD(z) ~ 1 only if the fitted sigma model actually tracks sigma_true.
    std = standardize(df, model, rescale=False)
    assert abs(nmad(std["z"].values) - 1.0) < 0.15


# ---------------------------------------------------------------------------
# regression: predictor_ranges must reflect the GAM's binned training table,
# not the raw predictor column, and standardize()'s sigma ceiling must catch
# whatever blow-up slips through anyway (see `fit_sigma_model` / `standardize`
# docstrings; observed on real lidar data: a right-skewed roughness
# predictor whose top quantile-bin centroid undershot the raw column max,
# letting `_clip()` pass real pixels through to spline territory the GAM
# never trained near, producing sigma ~10^6 m once exponentiated).
# ---------------------------------------------------------------------------

@requires_pygam
def test_predictor_ranges_reflect_binned_training_domain_not_raw_column():
    rng = np.random.default_rng(3)
    n = 6000
    bulk = rng.uniform(0.0, 2.0, n - 150)
    tail = rng.uniform(18.0, 20.0, 150)  # sparse, distant right tail
    roughness = np.concatenate([bulk, tail])
    rng.shuffle(roughness)
    dh = rng.normal(0, 1, n) * (0.1 + 0.01 * roughness)
    df = pd.DataFrame({
        "dh": dh, "roughness": roughness,
        "dom_class": pd.Categorical(["ground"] * n),
    })
    model = fit_sigma_model(df, predictors=["roughness"], factor_cols=[])
    raw_max = float(roughness.max())
    lo, hi = model.predictor_ranges["roughness"]
    # The old (buggy) implementation set this to the raw column max exactly;
    # for this right-skewed construction the binned-table max sits well
    # below it.
    assert hi < raw_max - 1.0
    assert lo >= float(roughness.min()) - 1e-9


@requires_pygam
def test_stratified_fit_predictor_ranges_populated():
    # Stratified branch (separate code path from the non-stratified one
    # above): predictor_ranges should still be finite, sane per-predictor
    # (min <= max) bounds, unioned across strata.
    df = _two_class_frame()
    model = fit_sigma_model(
        df, predictors=["slope", "roughness"], factor_cols=["dom_class"],
        stratify_by_class=True, strat_key="dom_class",
    )
    assert set(model.predictor_ranges.keys()) == {"slope", "roughness"}
    for lo, hi in model.predictor_ranges.values():
        assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


class _FakeSigmaModel:
    """Minimal duck-typed stand-in for HeteroscedasticSigmaModel.predict."""

    def __init__(self, sigmas):
        self._sigmas = np.asarray(sigmas, dtype=float)

    def predict(self, df):
        return self._sigmas


def test_standardize_sigma_ceiling_clips_blowup():
    n = 200
    sigmas = np.full(n, 0.1)
    sigmas[0] = 1e6  # a single pathological GAM-extrapolation-style blow-up
    rng = np.random.default_rng(4)
    dh = rng.normal(0, 0.1, n)
    df = pd.DataFrame({"dh": dh})
    std = standardize(df, _FakeSigmaModel(sigmas), rescale=False)
    assert std["sigma_hat"].values[0] < 1000.0  # nowhere near the raw 1e6
    assert std.attrs["n_sigma_ceiling_clipped"] == 1
    assert np.isclose(std["sigma_hat"].values[1], 0.1)  # untouched


def test_standardize_sigma_ceiling_inactive_for_well_behaved_predictions():
    rng = np.random.default_rng(5)
    n = 500
    sigmas = rng.uniform(0.05, 0.3, n)  # modest, realistic dynamic range
    dh = rng.normal(0, 1, n) * sigmas
    df = pd.DataFrame({"dh": dh})
    std = standardize(df, _FakeSigmaModel(sigmas), rescale=False)
    assert std.attrs["n_sigma_ceiling_clipped"] == 0


# ---------------------------------------------------------------------------
# regression: stratified double-count (#2) and index safety (#3)
# ---------------------------------------------------------------------------

def _two_class_frame(seed=0, n=12_000):
    rng = np.random.default_rng(seed)
    slope = rng.uniform(0, 30, n)
    cls = np.where(rng.random(n) < 0.5, "ground", "veg")
    # veg is intrinsically noisier than ground (a pure class effect)
    class_scale = np.where(cls == "veg", 0.8, 0.2)
    dh = rng.normal(0, 1, n) * class_scale
    return pd.DataFrame({
        "dh": dh, "slope": slope, "roughness": rng.uniform(0, 1, n),
        "dom_class": pd.Categorical(cls),
    })


@requires_pygam
def test_stratify_excludes_stratkey_from_offsets():
    # Fix #2: when stratifying by dom_class, dom_class must NOT also be an offset
    df = _two_class_frame()
    model = fit_sigma_model(
        df, predictors=["slope", "roughness"], factor_cols=["dom_class"],
        stratify_by_class=True, strat_key="dom_class",
    )
    assert "dom_class" not in model.factor_cols
    assert "dom_class" not in model.factor_offsets
    assert set(model.stratified.keys()) >= {"ground", "veg"}


@requires_pygam
def test_stratified_predict_not_double_counted():
    # σ predicted per class should track the true per-class scale (~0.2 / ~0.8),
    # not the squared/×2 value a double-applied offset would produce.
    df = _two_class_frame()
    model = fit_sigma_model(
        df, predictors=["slope", "roughness"], factor_cols=["dom_class"],
        stratify_by_class=True, strat_key="dom_class",
    )
    q = pd.DataFrame({"slope": [15.0, 15.0], "roughness": [0.5, 0.5],
                      "dom_class": pd.Categorical(["ground", "veg"])})
    s = model.predict(q)
    assert 0.1 < s[0] < 0.4      # ground ~0.2
    assert 0.5 < s[1] < 1.2      # veg ~0.8


@requires_pygam
def test_fit_sigma_model_handles_unreset_index():
    # Fix #3: a .loc[]-filtered (non-reset) frame must not misindex internally
    df = _two_class_frame()
    df = df[df["slope"] > 5.0]   # leaves a gappy, non-positional index
    model = fit_sigma_model(df, predictors=["slope"], factor_cols=[])
    out = model.predict(pd.DataFrame({"slope": [10.0]}))
    assert np.isfinite(out[0]) and out[0] > 0


def test_strip_id_union_indexing_is_consistent():
    # Fix #1 invariant: deriving strip ids ONCE over the full grid and indexing
    # subsets yields identical labels for shared pixels (unlike per-frame
    # derivation, which can relabel the same physical strip).
    gps = np.array([100, 100.1, 100.2, 200, 200.1, 300, 300.2, 100.3, 200.2])
    strip_all = derive_strip_ids(gps, gap_seconds=5.0)
    stable_idx = np.array([0, 3, 5, 7])
    subset_labels = strip_all[stable_idx]
    # gps[0]=100 and gps[7]=100.3 belong to the same physical strip, so
    # union-derived labels indexed by the subset must agree there ...
    assert subset_labels[0] == subset_labels[3]
    # ... and the three distinct strips stay distinct in the subset view
    assert len({subset_labels[0], subset_labels[1], subset_labels[2]}) == 3
    assert set(strip_all.tolist()) == {0, 1, 2}  # three physical strips detected


# ---------------------------------------------------------------------------
# point-cloud extraction (PDAL)
# ---------------------------------------------------------------------------

@requires_pdal
def test_extract_pointcloud_predictors_density(compare_laz_path):
    from topochange.heteroscedastic import extract_pointcloud_predictors
    import rasterio.transform as rt

    res = 25.0
    x0 = SYNTHETIC["x_offset"]
    y0 = SYNTHETIC["y_offset"]
    ext = SYNTHETIC["x_extent"]
    cols = int(np.ceil(ext / res))
    rows = int(np.ceil(SYNTHETIC["y_extent"] / res))
    transform = rt.from_origin(x0, y0 + SYNTHETIC["y_extent"], res, res)

    out = extract_pointcloud_predictors(
        str(compare_laz_path), transform, (rows, cols), None,
        class_spec=ClassificationSpec(),
    )
    assert out["density"].shape == (rows, cols)
    # every emitted point should land in some cell (grid covers the extent)
    assert out["density"].sum() > 0
    assert out["density"].sum() <= SYNTHETIC["n_points"] + 1


@requires_pdal
def test_extract_pointcloud_predictors_streaming_matches_batch(compare_laz_path):
    """Streaming aggregation is bit-for-bit identical to the batch path.

    Exercises the two things streaming has to get right: a grid *smaller* than
    the cloud extent (so most points are cropped / dropped) and accumulation
    across many chunks (chunk_size << n_points).
    """
    from topochange.heteroscedastic import extract_pointcloud_predictors
    from topochange.pdal_wrapper import pdal
    import rasterio.transform as rt

    if not pdal.Pipeline("[]").streamable:
        pytest.skip("native PDAL required for chunked streaming iteration")

    res = 25.0
    x0, y0 = SYNTHETIC["x_offset"], SYNTHETIC["y_offset"]
    # Grid covers only the central 300x300 m of the 500x500 m cloud.
    win, off = 300.0, 100.0
    rows = cols = int(win / res)
    transform = rt.from_origin(x0 + off, y0 + off + win, res, res)

    rng = np.random.default_rng(0)
    slope = rng.uniform(0, 40, (rows, cols))
    aspect = rng.uniform(0, 360, (rows, cols))
    kw = dict(
        class_spec=ClassificationSpec(),
        sensor_altitude=1500.0,
        normal_from_slope_aspect=(slope, aspect),
    )

    batch = extract_pointcloud_predictors(
        str(compare_laz_path), transform, (rows, cols), None, stream=False, **kw
    )
    assert 0 < batch["density"].sum() < SYNTHETIC["n_points"]  # cropping happened

    for chunk in (10_000, 1_000):  # ~5 and ~50 chunks over the cropped subset
        strm = extract_pointcloud_predictors(
            str(compare_laz_path), transform, (rows, cols), None,
            stream=True, chunk_size=chunk, **kw,
        )
        assert set(strm) == set(batch)
        for k in batch:
            a = np.asarray(batch[k], dtype=float)
            b = np.asarray(strm[k], dtype=float)
            assert not (np.isnan(a) ^ np.isnan(b)).any(), f"{k}: NaN pattern differs"
            both_nan = np.isnan(a) & np.isnan(b)
            np.testing.assert_array_equal(
                np.where(both_nan, 0.0, a), np.where(both_nan, 0.0, b),
                err_msg=f"{k}: streaming (chunk={chunk}) != batch",
            )


@requires_pdal
def test_extract_pointcloud_predictors_point_source_id_absent(compare_laz_path):
    """The shared conftest synthetic cloud does carry a PointSourceId
    dimension (LAS point format 1 always has the field), but conftest never
    sets it, so PDAL's writer defaults every point to 0: a single constant
    id, i.e. present but uninformative. Every covered cell should read 0
    (not -1: -1 is reserved for cells with *no point-cloud coverage at
    all*), and the field must still be correctly judged uninformative."""
    from topochange.heteroscedastic import extract_pointcloud_predictors
    import rasterio.transform as rt

    res = 25.0
    x0, y0 = SYNTHETIC["x_offset"], SYNTHETIC["y_offset"]
    cols = int(np.ceil(SYNTHETIC["x_extent"] / res))
    rows = int(np.ceil(SYNTHETIC["y_extent"] / res))
    transform = rt.from_origin(x0, y0 + SYNTHETIC["y_extent"], res, res)

    out = extract_pointcloud_predictors(
        str(compare_laz_path), transform, (rows, cols), None,
        class_spec=ClassificationSpec(),
    )
    assert "point_source_id" in out
    assert out["point_source_id"].shape == (rows, cols)
    assert not is_point_source_id_informative(out["point_source_id"])
    covered = out["density"] > 0
    assert covered.any()  # sanity: the grid does overlap the cloud
    assert (out["point_source_id"][covered] == 0).all()
    assert (out["point_source_id"][~covered] == -1).all()


def _create_synthetic_laz_with_strips(
    filepath,
    *,
    n_points: int = 20_000,
    x_offset: float = 500_000.0,
    y_offset: float = 4_000_000.0,
    x_extent: float = 500.0,
    y_extent: float = 500.0,
    epsg: int = 32610,
    gps_time_base: float = 7.9e8,
    seed: int = 7,
):
    """Synthetic LAZ split into two spatially-separated PointSourceId strips:
    ``x < x_extent/2`` gets source id 101, ``x >= x_extent/2`` gets 202, so
    the per-cell majority-vote id is unambiguous away from the boundary
    column. GpsTime is drawn from one short, gap-free interval (no jumps
    > a few seconds) so it does *not* independently suggest two strips,
    keeping the PointSourceId-vs-GPS-gap distinction meaningful in tests
    that compare the two strip_id sources.

    Mirrors conftest.py's ``_create_synthetic_laz`` (same field set, writer
    pipeline, and PDAL access pattern) with a PointSourceId dimension added.
    """
    from topochange.pdal_wrapper import pdal

    rng = np.random.default_rng(seed)
    x = rng.uniform(0, x_extent, n_points)
    y = rng.uniform(0, y_extent, n_points)
    z = 1000.0 + 5.0 * np.sin(x / 40.0) * np.cos(y / 40.0) + rng.normal(0, 0.05, n_points)
    psid = np.where(x < x_extent / 2.0, 101, 202).astype(np.uint16)
    gps_time = gps_time_base + rng.uniform(0, 3600.0, n_points)

    dtype = np.dtype([
        ("X", "f8"), ("Y", "f8"), ("Z", "f8"),
        ("Intensity", "u2"), ("ReturnNumber", "u1"), ("NumberOfReturns", "u1"),
        ("Classification", "u1"), ("PointSourceId", "u2"), ("GpsTime", "f8"),
    ])
    arr = np.zeros(n_points, dtype=dtype)
    arr["X"] = x + x_offset
    arr["Y"] = y + y_offset
    arr["Z"] = z
    arr["ReturnNumber"] = 1
    arr["NumberOfReturns"] = 1
    arr["Classification"] = 2
    arr["PointSourceId"] = psid
    arr["GpsTime"] = gps_time

    pipeline_spec = {
        "pipeline": [
            {
                "type": "writers.las",
                "filename": str(filepath),
                "a_srs": f"EPSG:{epsg}",
                "compression": "laszip",
                "minor_version": 4,
                "dataformat_id": 1,
                "offset_x": x_offset,
                "offset_y": y_offset,
                "offset_z": 0.0,
                "scale_x": 0.001,
                "scale_y": 0.001,
                "scale_z": 0.001,
            }
        ]
    }
    pipeline = pdal.Pipeline(json.dumps(pipeline_spec), arrays=[arr])
    pipeline.execute()


@requires_pdal
def test_extract_pointcloud_predictors_point_source_id_dominant_vote(tmp_path):
    from topochange.heteroscedastic import extract_pointcloud_predictors
    import rasterio.transform as rt

    laz_path = tmp_path / "strips.laz"
    _create_synthetic_laz_with_strips(laz_path)

    res = 25.0
    x0, y0, ext = 500_000.0, 4_000_000.0, 500.0
    cols = rows = int(ext / res)
    transform = rt.from_origin(x0, y0 + ext, res, res)

    out = extract_pointcloud_predictors(
        str(laz_path), transform, (rows, cols), None, class_spec=ClassificationSpec(),
    )
    psid_grid = out["point_source_id"]
    assert psid_grid.shape == (rows, cols)
    assert is_point_source_id_informative(psid_grid)
    seen = set(np.unique(psid_grid).tolist())
    assert seen <= {-1, 101, 202} and {101, 202} <= seen

    # left half of the grid should be dominated by strip 101, right half by
    # 202; stay one column clear of the x_extent/2 boundary, where a cell
    # can legitimately straddle both strips.
    boundary_col = cols // 2
    left = psid_grid[:, : boundary_col - 1]
    right = psid_grid[:, boundary_col + 1:]
    assert (left[left >= 0] == 101).all()
    assert (right[right >= 0] == 202).all()


@requires_pdal
def test_extract_pointcloud_predictors_point_source_id_streaming_matches_batch(tmp_path):
    from topochange.heteroscedastic import extract_pointcloud_predictors
    from topochange.pdal_wrapper import pdal
    import rasterio.transform as rt

    if not pdal.Pipeline("[]").streamable:
        pytest.skip("native PDAL required for chunked streaming iteration")

    laz_path = tmp_path / "strips_stream.laz"
    _create_synthetic_laz_with_strips(laz_path, n_points=40_000, seed=11)

    res = 20.0
    x0, y0, ext = 500_000.0, 4_000_000.0, 500.0
    cols = rows = int(ext / res)
    transform = rt.from_origin(x0, y0 + ext, res, res)

    batch = extract_pointcloud_predictors(
        str(laz_path), transform, (rows, cols), None,
        class_spec=ClassificationSpec(), stream=False,
    )
    for chunk in (5_000, 700):
        strm = extract_pointcloud_predictors(
            str(laz_path), transform, (rows, cols), None,
            class_spec=ClassificationSpec(), stream=True, chunk_size=chunk,
        )
        np.testing.assert_array_equal(
            batch["point_source_id"], strm["point_source_id"],
            err_msg=f"point_source_id: streaming (chunk={chunk}) != batch",
        )


# ---------------------------------------------------------------------------
# end-to-end smoke test (terrain-only path, no PDAL; pygam-gated)
# ---------------------------------------------------------------------------

def _write_synthetic_case(tmpdir):
    """Write a synthetic dh + terrain rasters with a KNOWN heteroscedastic
    structure and return the file paths, CRS transform, and an FOI polygon.

    Construction
    ------------
    * A smooth, spatially-correlated standardised field ``zf`` (sum of a few
      low-frequency sinusoids + light white noise) so its variogram has real
      range to fit.
    * ``slope`` ramps across the grid; ``σ_true = 0.15 + 0.03·slope``.
    * ``dh = σ_true · zf``, so ``dh/σ`` recovers ``zf`` and the σ model must
      learn the slope dependence.
    """
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    rng = np.random.default_rng(42)
    rows = cols = 120
    res = 10.0
    x0, y0 = 500_000.0, 4_000_000.0
    transform = from_origin(x0, y0 + rows * res, res, res)
    crs = CRS.from_epsg(32610)

    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    # smooth correlated standardised field
    zf = (np.sin(xx / 12.0) + np.cos(yy / 15.0)
          + 0.5 * np.sin((xx + yy) / 20.0) + rng.normal(0, 0.1, (rows, cols)))
    slope = 2.0 + 0.30 * yy                     # ramps 2 -> ~38 degrees
    aspect = np.full((rows, cols), 90.0)
    roughness = 0.5 + 0.01 * xx
    sigma_true = 0.15 + 0.03 * slope
    dh = (sigma_true * zf).astype(np.float64)
    stable = np.ones((rows, cols), dtype=np.uint8)

    paths = {}
    for name, arr, dtype, nodata in [
        ("dh", dh, "float32", np.nan),
        ("slope", slope, "float32", np.nan),
        ("aspect", aspect, "float32", np.nan),
        ("roughness", roughness, "float32", np.nan),
        ("stable", stable, "uint8", 0),
    ]:
        p = str(tmpdir / f"{name}.tif")
        prof = {
            "driver": "GTiff", "height": rows, "width": cols, "count": 1,
            "dtype": dtype, "transform": transform, "crs": crs, "nodata": nodata,
        }
        with rasterio.open(p, "w", **prof) as dst:
            dst.write(arr.astype(dtype), 1)
        paths[name] = p

    # FOI polygon covering the central quarter of the grid (in map coords)
    cx0 = x0 + 30 * res
    cx1 = x0 + 90 * res
    cy_top = y0 + rows * res - 30 * res
    cy_bot = y0 + rows * res - 90 * res
    foi = Polygon([(cx0, cy_bot), (cx1, cy_bot), (cx1, cy_top), (cx0, cy_top)])
    return paths, foi


@requires_pygam
def test_pipeline_smoke_isotropic(tmp_path):
    from topochange.heteroscedastic import run_heteroscedastic_pipeline

    paths, foi = _write_synthetic_case(tmp_path)
    result = run_heteroscedastic_pipeline(
        dh_path=paths["dh"],
        las_path=None,
        slope_path=paths["slope"],
        aspect_path=paths["aspect"],
        roughness_path=paths["roughness"],
        stable_mask_path=paths["stable"],
        foi_geometries={"aoi": foi},
        predictors=["slope", "roughness"],
        include_strip_id=False,
        variogram_directions=None,      # isotropic
        variogram_components=["spherical"],
        fit_anisotropy=False,
        areal_method="mc",
        mc_realizations=100,
        seed=0,
    )

    # σ raster: finite, positive, and larger where slope is larger (bottom rows)
    sig = result["sigma_raster"]
    assert np.isfinite(sig).mean() > 0.9
    assert np.nanmin(sig) > 0
    top = np.nanmedian(sig[:20, :])
    bot = np.nanmedian(sig[-20:, :])
    assert bot > top  # σ grows with slope

    # standardisation should be roughly unit-scale
    assert abs(result["diagnostics"]["nmad"] - 1.0) < 0.3

    # correlogram is stationary with a positive sill
    vgm = result["variogram_model"]
    assert vgm.total_sill() > 0
    assert vgm.anisotropy_ratio == 1.0

    # areal σ over the FOI is finite, positive, and below the point-σ scale
    u = result["foi_uncertainty"]["aoi"]
    assert np.isfinite(u["sigma_area"]) and u["sigma_area"] > 0
    assert u["sigma_area"] < float(np.nanmax(sig))
    assert u["n_pixels"] > 100


@requires_pygam
def test_pipeline_smoke_anisotropic(tmp_path):
    from topochange.heteroscedastic import run_heteroscedastic_pipeline

    paths, foi = _write_synthetic_case(tmp_path)
    result = run_heteroscedastic_pipeline(
        dh_path=paths["dh"],
        las_path=None,
        slope_path=paths["slope"],
        aspect_path=paths["aspect"],
        roughness_path=paths["roughness"],
        stable_mask_path=paths["stable"],
        foi_geometries={"aoi": foi},
        predictors=["slope", "roughness"],
        include_strip_id=False,
        variogram_directions=[(0, 22.5), (45, 22.5), (90, 22.5), (135, 22.5)],
        variogram_components=["spherical", "spherical"],
        fit_anisotropy=True,
        areal_method="pairwise",
        n_pairs=50_000,
        seed=0,
    )
    emp = result["empirical_variogram"]
    assert set(emp["direction"].unique()) <= {0.0, 45.0, 90.0, 135.0}
    vgm = result["variogram_model"]
    assert 0.1 <= vgm.anisotropy_ratio <= 10.0
    assert vgm.total_sill() > 0
    u = result["foi_uncertainty"]["aoi"]
    assert np.isfinite(u["sigma_area"]) and u["sigma_area"] > 0


# ---------------------------------------------------------------------------
# run_heteroscedastic_pipeline: strip_id_source (PointSourceId vs GPS-gap)
# ---------------------------------------------------------------------------

def _create_synthetic_laz_gps_gap_no_psid(
    filepath,
    *,
    n_points: int = 20_000,
    x_offset: float = 500_000.0,
    y_offset: float = 4_000_000.0,
    x_extent: float = 500.0,
    y_extent: float = 500.0,
    epsg: int = 32610,
    seed: int = 13,
):
    """Synthetic LAZ with two GPS-time clusters (a clean gap >> 5 s) and NO
    PointSourceId dimension at all, the mirror image of
    ``_create_synthetic_laz_with_strips`` above. Used to test the
    ``strip_id_source='point_source_id'`` -> warn -> fall back to
    ``derive_strip_ids`` (GPS-gap) path when the field genuinely isn't there.
    """
    from topochange.pdal_wrapper import pdal

    rng = np.random.default_rng(seed)
    x = rng.uniform(0, x_extent, n_points)
    y = rng.uniform(0, y_extent, n_points)
    z = 1000.0 + 5.0 * np.sin(x / 40.0) * np.cos(y / 40.0) + rng.normal(0, 0.05, n_points)
    half = n_points // 2
    gps_time = np.concatenate([
        7.9e8 + rng.uniform(0, 100.0, half),
        7.9e8 + 500.0 + rng.uniform(0, 100.0, n_points - half),
    ])

    dtype = np.dtype([
        ("X", "f8"), ("Y", "f8"), ("Z", "f8"),
        ("Intensity", "u2"), ("ReturnNumber", "u1"), ("NumberOfReturns", "u1"),
        ("Classification", "u1"), ("GpsTime", "f8"),
    ])
    arr = np.zeros(n_points, dtype=dtype)
    arr["X"] = x + x_offset
    arr["Y"] = y + y_offset
    arr["Z"] = z
    arr["ReturnNumber"] = 1
    arr["NumberOfReturns"] = 1
    arr["Classification"] = 2
    arr["GpsTime"] = gps_time

    pipeline_spec = {
        "pipeline": [
            {
                "type": "writers.las",
                "filename": str(filepath),
                "a_srs": f"EPSG:{epsg}",
                "compression": "laszip",
                "minor_version": 4,
                "dataformat_id": 1,
                "offset_x": x_offset,
                "offset_y": y_offset,
                "offset_z": 0.0,
                "scale_x": 0.001,
                "scale_y": 0.001,
                "scale_z": 0.001,
            }
        ]
    }
    pdal.Pipeline(json.dumps(pipeline_spec), arrays=[arr]).execute()


@requires_pygam
@requires_pdal
def test_pipeline_strip_id_source_auto_prefers_point_source_id(tmp_path):
    from topochange.heteroscedastic import run_heteroscedastic_pipeline

    paths, _ = _write_synthetic_case(tmp_path)  # 120x120 @ 10 m, origin (500000, 4000000)
    laz_path = tmp_path / "strips.laz"
    _create_synthetic_laz_with_strips(
        laz_path, n_points=40_000, seed=3,
        x_offset=500_000.0, y_offset=4_000_000.0, x_extent=1200.0, y_extent=1200.0,
    )

    result = run_heteroscedastic_pipeline(
        dh_path=paths["dh"],
        las_path=str(laz_path),
        slope_path=paths["slope"],
        aspect_path=paths["aspect"],
        roughness_path=paths["roughness"],
        stable_mask_path=paths["stable"],
        predictors=["slope", "roughness"],
        factor_cols=["strip_id"],
        include_strip_id=True,
        strip_id_source="auto",
        variogram_directions=None,
        variogram_components=["spherical"],
        fit_anisotropy=False,
        areal_method="mc",
        mc_realizations=50,
        seed=0,
    )
    assert result["diagnostics"]["strip_id_source_used"] == "point_source_id"
    strip_ids = set(result["standardized_frame"]["strip_id"].astype(int).unique().tolist())
    # both physical strips must be detected; -1 (a handful of cells with no
    # point-cloud coverage, expected at ~40k points over a 1200x1200 m grid)
    # is allowed but not required.
    assert {0, 1} <= strip_ids <= {-1, 0, 1}


@requires_pygam
@requires_pdal
def test_pipeline_strip_id_source_gps_gap_explicit_override(tmp_path):
    """Even though PointSourceId is available and informative,
    strip_id_source='gps_gap' must force the GPS-time-heuristic path."""
    from topochange.heteroscedastic import run_heteroscedastic_pipeline

    paths, _ = _write_synthetic_case(tmp_path)
    laz_path = tmp_path / "strips.laz"
    _create_synthetic_laz_with_strips(
        laz_path, n_points=40_000, seed=3,
        x_offset=500_000.0, y_offset=4_000_000.0, x_extent=1200.0, y_extent=1200.0,
    )

    result = run_heteroscedastic_pipeline(
        dh_path=paths["dh"],
        las_path=str(laz_path),
        slope_path=paths["slope"],
        aspect_path=paths["aspect"],
        roughness_path=paths["roughness"],
        stable_mask_path=paths["stable"],
        predictors=["slope", "roughness"],
        factor_cols=["strip_id"],
        include_strip_id=True,
        strip_id_source="gps_gap",
        variogram_directions=None,
        variogram_components=["spherical"],
        fit_anisotropy=False,
        areal_method="mc",
        mc_realizations=50,
        seed=0,
    )
    assert result["diagnostics"]["strip_id_source_used"] == "gps_gap"


@requires_pygam
@requires_pdal
def test_pipeline_strip_id_source_point_source_id_falls_back_with_warning(tmp_path):
    """strip_id_source='point_source_id' with a cloud that has no
    PointSourceId dimension should warn and fall back to the GPS-gap
    heuristic, not error."""
    from topochange.heteroscedastic import run_heteroscedastic_pipeline

    paths, _ = _write_synthetic_case(tmp_path)
    laz_path = tmp_path / "no_psid.laz"
    _create_synthetic_laz_gps_gap_no_psid(
        laz_path, n_points=40_000, seed=13,
        x_offset=500_000.0, y_offset=4_000_000.0, x_extent=1200.0, y_extent=1200.0,
    )

    with pytest.warns(UserWarning, match="PointSourceId"):
        result = run_heteroscedastic_pipeline(
            dh_path=paths["dh"],
            las_path=str(laz_path),
            slope_path=paths["slope"],
            aspect_path=paths["aspect"],
            roughness_path=paths["roughness"],
            stable_mask_path=paths["stable"],
            predictors=["slope", "roughness"],
            factor_cols=["strip_id"],
            include_strip_id=True,
            strip_id_source="point_source_id",
            variogram_directions=None,
            variogram_components=["spherical"],
            fit_anisotropy=False,
            areal_method="mc",
            mc_realizations=50,
            seed=0,
        )
    assert result["diagnostics"]["strip_id_source_used"] == "gps_gap"


def test_pipeline_invalid_strip_id_source_raises(tmp_path):
    """Validated before any point-cloud / GAM work, so this needs neither
    PDAL nor pygam to be installed."""
    from topochange.heteroscedastic import run_heteroscedastic_pipeline

    paths, _ = _write_synthetic_case(tmp_path)
    with pytest.raises(ValueError, match="strip_id_source"):
        run_heteroscedastic_pipeline(
            dh_path=paths["dh"],
            las_path=None,
            slope_path=paths["slope"],
            aspect_path=paths["aspect"],
            roughness_path=paths["roughness"],
            stable_mask_path=paths["stable"],
            predictors=["slope", "roughness"],
            include_strip_id=True,
            strip_id_source="bogus",
        )


# ---------------------------------------------------------------------------
# per-cell dominant PointSourceId vote (pure numpy -- no pdal needed)
# ---------------------------------------------------------------------------

def test_dominant_id_per_cell_majority_ties_and_empty_cells():
    from topochange.heteroscedastic import _dominant_id_per_cell
    cell = np.array([0, 0, 0, 1, 1, 2])
    pid = np.array([7, 7, 3, 9, 5, 4])
    out = _dominant_id_per_cell(cell, pid, n_cells=4)
    # cell 0: 7 wins 2-1; cell 1: exact tie 5 vs 9 -> smaller id; cell 3: empty
    assert out.tolist() == [7, 5, 4, -1]


def test_dominant_id_per_cell_empty_input():
    from topochange.heteroscedastic import _dominant_id_per_cell
    out = _dominant_id_per_cell(np.array([], dtype=np.int64),
                                np.array([], dtype=np.int64), n_cells=3)
    assert out.tolist() == [-1, -1, -1]


def test_dominant_id_streaming_key_counts_matches_batch():
    """The streaming (key, count) merge must agree with the batch vote."""
    from topochange.heteroscedastic import (
        _dominant_id_per_cell, _dominant_id_from_key_counts,
    )
    rng = np.random.default_rng(3)
    cell = rng.integers(0, 50, 5000)
    pid = rng.integers(0, 8, 5000)
    key = cell.astype(np.int64) * 65536 + pid.astype(np.int64)
    uniq, counts = np.unique(key, return_counts=True)
    batch = _dominant_id_per_cell(cell, pid, 50)
    streamed = _dominant_id_from_key_counts(uniq, counts, 50)
    assert np.array_equal(batch, streamed)
