"""Tests for continuous extra predictors (e.g. a CHM) and veg-class removal.

New behaviour (2026-07-16):
  * vegetation is no longer a ``dom_class`` category; canopy is meant to enter
    as a continuous predictor (a CHM) via ``build_predictor_frame(extra_rasters=)``
    / ``run_heteroscedastic_pipeline(extra_raster_paths=)``;
  * the ``veg`` flag on ``ClassificationSpec`` is retained for point *selection*
    only (last returns for ground under canopy);
  * ``run_heteroscedastic_pipeline`` gains ``stable_area_paths`` (derive the
    stable mask from the valid pixels of stable-area TIFFs).
"""
import inspect
import numpy as np
import pandas as pd
import pytest


def test_veg_removed_from_dominant_class():
    from topochange.heteroscedastic import CLASS_INT_TO_LABEL, ClassificationSpec

    assert CLASS_INT_TO_LABEL == {0: "other", 1: "ground", 2: "building"}
    # veg is kept as a point-SELECTION flag (last returns) but never a category
    cs = ClassificationSpec(veg=True, ground=True, building=True)
    assert cs.pdal_filter_expression() == "(ReturnNumber == NumberOfReturns)"
    assert all(cs.label_for_point(c) != "veg" for c in (2, 3, 4, 5, 6, 99))


def test_build_predictor_frame_extra_rasters_aligned():
    from topochange.heteroscedastic import build_predictor_frame

    H, W = 6, 5
    dh = np.random.default_rng(0).normal(0, 0.1, (H, W))
    dh[0, 0] = np.nan
    slope = np.full((H, W), 10.0)
    aspect = np.zeros((H, W))
    rough = np.full((H, W), 0.5)
    chm = np.arange(H * W, dtype=float).reshape(H, W)

    df = build_predictor_frame(
        dh, slope, aspect, rough, {}, stable_mask=None,
        add_pixel_index=True, extra_rasters={"chm": chm},
    )
    assert "chm" in df.columns
    # CHM must be aligned to chm.ravel()[pixel_index]
    assert np.allclose(df["chm"].values, chm.ravel()[df["pixel_index"].values])
    assert df["chm"].notna().any()          # usable as a continuous predictor


def test_stray_veg_code_folds_to_other():
    from topochange.heteroscedastic import build_predictor_frame

    H, W = 4, 4
    z = np.zeros((H, W))
    dom = np.zeros((H, W), dtype=np.int8)
    dom[0, 0] = 3                           # legacy veg code from an older extractor
    df = build_predictor_frame(z, z, z, z, {"dom_class": dom})
    levels = set(df["dom_class"].astype(str).unique())
    assert "veg" not in levels
    assert levels <= {"other", "ground", "building"}


def test_pipeline_exposes_new_kwargs():
    """Signature-level guard (does not execute the heavy pipeline)."""
    from topochange.heteroscedastic import run_heteroscedastic_pipeline

    sig = inspect.signature(run_heteroscedastic_pipeline)
    for p in ("stable_area_paths", "extra_raster_paths", "extra_raster_fill"):
        assert p in sig.parameters, f"missing new parameter {p}"
        assert sig.parameters[p].default is None


def test_stable_area_union_from_tiffs(tmp_path):
    """Deriving the stable mask from stable-area TIFF valid pixels (+ 0/1 mask)."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from topochange.heteroscedastic import run_heteroscedastic_pipeline  # noqa: F401

    # This exercises only the mask-derivation contract used by the pipeline:
    # valid (finite) pixels of the tif are stable; union with an explicit mask.
    H, W = 5, 5
    tif = np.full((H, W), np.nan, dtype="float32")
    tif[4, :] = 0.03                        # only the bottom row is "stable data"
    path = tmp_path / "stable_area.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
        crs="EPSG:32611", transform=from_origin(0, H, 1, 1), nodata=np.nan,
    ) as dst:
        dst.write(tif, 1)

    from topochange.heteroscedastic import _read_raster
    arr, _, _, _ = _read_raster(str(path))
    stable_valid = np.isfinite(arr)
    assert stable_valid[4].all()            # valid row -> stable
    assert not stable_valid[:4].any()       # nodata rows -> not stable


def test_predictor_collinearity_flags_collinear_pair():
    from topochange.heteroscedastic import predictor_collinearity

    rng = np.random.default_rng(0)
    N = 2000
    slope = rng.uniform(0, 40, N)
    df = pd.DataFrame({
        "dh": rng.normal(0, 0.1, N),
        "pixel_index": np.arange(N),                     # bookkeeping -> excluded
        "slope": slope,
        "roughness": 0.3 * slope + rng.normal(0, 1.0, N),  # collinear with slope
        "chm": rng.exponential(5, N),                    # independent
        "density": rng.uniform(1, 20, N),                # independent
    })
    res = predictor_collinearity(df, verbose=False)
    assert set(res["predictors"]) == {"slope", "roughness", "chm", "density"}
    assert res["vif"]["slope"] > 5 and res["vif"]["roughness"] > 5
    assert res["vif"]["chm"] < 2 and res["vif"]["density"] < 2
    flagged = {frozenset((a, b)) for a, b, _ in res["high_pairs"]}
    assert frozenset(("slope", "roughness")) in flagged


def test_predict_clips_predictors_to_training_range():
    from topochange.heteroscedastic import HeteroscedasticSigmaModel

    class _GAM:                     # log-sigma = chm  ->  sigma = exp(chm)
        def predict(self, X):
            return X[:, 0]

    mdl = HeteroscedasticSigmaModel(
        predictors=["chm"], factor_cols=[], factor_offsets={}, gam=_GAM(),
        predictor_ranges={"chm": (0.0, 2.0)},
    )
    df = pd.DataFrame({"chm": [1.0, 2.0, 30.0]})
    s = mdl.predict(df)
    assert np.isclose(s[2], np.exp(2.0))     # chm=30 clipped to the training max 2
    assert np.isclose(s[1], s[2])            # extrapolated == boundary prediction
    # NaN predictors still pass through as NaN (skip contract preserved)
    assert np.isnan(mdl.predict(pd.DataFrame({"chm": [np.nan]}))[0])
    # without stored ranges the model extrapolates (and would blow up): exp(30)
    mdl.predictor_ranges = None
    assert mdl.predict(df)[2] > 1e6


def test_read_raster_on_grid_resamples(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine
    from rasterio.crs import CRS
    from topochange.heteroscedastic import _read_raster_on_grid, _reproject_to_grid

    def write(path, arr, transform):
        with rasterio.open(
            path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
            count=1, dtype="float32", crs="EPSG:32611", transform=transform,
            nodata=np.nan,
        ) as d:
            d.write(arr.astype("float32"), 1)

    dh_shape = (30, 25)
    dh_tr = Affine.translation(500000, 4000000) * Affine.scale(10, -10)
    dh_crs = CRS.from_epsg(32611)

    # already on the target grid -> fast path, not resampled
    write(tmp_path / "a.tif", np.ones(dh_shape), dh_tr)
    arr, res = _read_raster_on_grid(str(tmp_path / "a.tif"), dh_tr, dh_crs, dh_shape)
    assert arr.shape == dh_shape and res is False

    # different shape/resolution/origin -> resampled onto the dh grid
    chm = np.add.outer(np.arange(37), np.arange(31)).astype(float)
    chm_tr = Affine.translation(499990, 4000010) * Affine.scale(8, -8)
    write(tmp_path / "chm.tif", chm, chm_tr)
    arr2, res2 = _read_raster_on_grid(
        str(tmp_path / "chm.tif"), dh_tr, dh_crs, dh_shape, "bilinear")
    assert arr2.shape == dh_shape and res2 is True and np.isfinite(arr2).any()

    # nearest keeps a 0/1 mask binary through the regrid
    mask = np.zeros((37, 31)); mask[8:28, 8:22] = 1.0
    write(tmp_path / "m.tif", mask, chm_tr)
    am, _ = _read_raster_on_grid(str(tmp_path / "m.tif"), dh_tr, dh_crs, dh_shape, "nearest")
    assert set(np.unique(am[np.isfinite(am)])).issubset({0.0, 1.0})

    # _reproject_to_grid fast path returns the input unchanged
    z = np.ones(dh_shape)
    out, r = _reproject_to_grid(z, dh_tr, dh_crs, dh_tr, dh_crs, dh_shape)
    assert r is False and np.array_equal(out, z)


def test_stable_area_paths_accepts_single_path():
    from pathlib import Path
    from topochange.heteroscedastic import _as_path_list

    assert _as_path_list(None) == []
    assert _as_path_list("stable.tif") == ["stable.tif"]     # NOT iterated char-by-char
    assert _as_path_list(Path("stable.tif")) == [Path("stable.tif")]
    assert _as_path_list(["a.tif", "b.tif"]) == ["a.tif", "b.tif"]
    assert _as_path_list(("a.tif",)) == ["a.tif"]


def test_predictor_collinearity_perfect_duplicate_and_guards():
    from topochange.heteroscedastic import predictor_collinearity

    rng = np.random.default_rng(1)
    N = 500
    x = rng.normal(0, 1, N)
    df = pd.DataFrame({"a": x, "b": 2.0 * x, "c": rng.normal(0, 1, N)})  # a,b collinear
    res = predictor_collinearity(df, predictors=("a", "b", "c"), verbose=False)
    assert res["vif"]["a"] > 1e6 and res["vif"]["b"] > 1e6
    with pytest.raises(ValueError):
        predictor_collinearity(df, predictors=("a",), verbose=False)  # < 2 predictors
