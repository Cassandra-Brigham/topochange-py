"""P0-1 envelope-coverage validation on grf_dataset.

Run in your FULL environment (package installed; rasterio + numba available).

Compares the sigma_A confidence-envelope calibration of:
  * LEGACY : single-model parametric bootstrap   (calibrate=False)
  * P0-1    : calibrated model-averaged ensemble  (calibrate=True)

Over the benchmark polygons it measures the fraction of cases whose
ANALYTIC-true correlated sigma_A falls inside the reported [min, max] envelope
(nominal 68%).  Well-calibrated -> ~0.68; the legacy single-model envelope is
expected to be too narrow.

Notebook use
------------
    import sys; sys.path.insert(0, "tests")
    import validate_p1_coverage as v1
    files = v1.list_datasets("Spherical_r100_*_n0_*") + v1.list_datasets("Exponential_r100_*_n0_*")
    df = v1.run_coverage(files=files, detrend=0)          # or patterns=[...], or n_fields=40
"""
import os, re, glob, math, random
import numpy as np
import geopandas as gpd

from topochange import (
    RasterDataHandler, SingleVariogram, RegionalUncertaintyEstimator,
    CompositeVariogramModel,
)

HERE = os.path.dirname(__file__)
DATA = os.environ.get("TOPO_GRF_DIR", os.path.join(HERE, "..", "synthetic_benchmark", "grf_dataset"))
POLY_DIR = os.path.join(HERE, "..", "synthetic_benchmark", "polygons")
SEED = 12345
N_PAIRS = 25_000
FAMILY = {"Spherical": "spherical", "Exponential": "exponential",
          "Gaussian": "gaussian", "Matern": "matern"}

def list_datasets(pattern="*", data=None):
    """Sorted absolute paths of grf tifs matching a glob (families fittable by topochange)."""
    data = data or DATA
    if not pattern.endswith(".tif"):
        pattern += ".tif"
    return [f for f in sorted(glob.glob(os.path.join(data, pattern)))
            if parse_truth(os.path.basename(f)) is not None]

def parse_truth(fname):
    fam = fname.split("_")[0]
    if fam not in FAMILY:
        return None
    m1, m2 = re.search(r"_r(\d+)_", fname), re.search(r"_s([0-9\-]+)_", fname)
    if not (m1 and m2):
        return None
    s = m2.group(1).replace("-", ".")
    if s.startswith("."): s = "0" + s
    return FAMILY[fam], int(m1.group(1)), float(s)

def true_model(family, rng_, sill):
    """Composite model for the PRESCRIBED truth.

    The dataset generator (see ``_gs_model``) interprets the filename ``r``
    as the 95% EFFECTIVE range, whereas topochange's model functions take
    the family's native range parameter.  Convert via the registry's
    practical_range_factor (spherical 1.0, exponential/matern ~3.0,
    gaussian ~1.73) so the truth model matches the generated fields.
    """
    from topochange.variogram_models import MODEL_REGISTRY

    prf = float(MODEL_REGISTRY.get_model(family).practical_range_factor)
    m = CompositeVariogramModel([family], include_nugget=False)
    p = [sill, float(rng_) / prf] + ([1.5] if family == "matern" else [])
    m.set_params(np.asarray(p, float))
    return m

def load_polygons():
    geoms = {}
    for shp in sorted(glob.glob(os.path.join(POLY_DIR, "*.shp"))):
        g = gpd.read_file(shp)
        geoms[os.path.splitext(os.path.basename(shp))[0]] = g.geometry.unary_union
    return geoms

def run_one(path, poly, calibrate, detrend=0, max_samples=1000, n_boot=150):
    """Fit the workflow for one field. max_samples/n_boot trade speed vs precision;
    the parametric bootstrap is O(max_samples^3) per realization, so keep
    max_samples modest for a fast sweep and raise it for the final run."""
    rdh = RasterDataHandler(path, unit="m", resolution=1.0); rdh.load_raster()
    sv = SingleVariogram(rdh)
    sv.compute_empirical_variogram(area_side=1.0, samples_per_area=1.0, max_samples=max_samples,
                                   bin_width=2.0, seed=SEED, return_sample=True,
                                   detrend_order=detrend)
    sv.fit_model(max_components=1, criterion="aicc", seed=SEED)
    if calibrate:
        # pre-build the calibrated ensemble with a controlled realization budget so
        # the estimator reuses it instead of rebuilding at the default 300.
        sv.calibrated_parameter_ensemble(n_realizations=n_boot, seed=SEED)
    else:
        # legacy single-model realization envelope
        sv.parametric_bootstrap_parameters(n_realizations=n_boot, seed=SEED)
    return RegionalUncertaintyEstimator(rdh, sv, area_of_interest=poly, calibrate=calibrate)

def run_coverage(files=None, patterns=None, n_fields=24, detrend=0, polygons=None,
                 max_samples=1000, n_boot=150, truth_fn=None, seed=SEED, verbose=True):
    """Measure legacy vs calibrated 68% coverage over chosen fields x polygons.

    Choose datasets with `files=[...paths...]`, `patterns=["Spherical_r100_*", ...]`,
    or leave both None to sample `n_fields` at random across fittable families.
    Returns a pandas DataFrame of per-(field, polygon, mode) results.
    """
    random.seed(seed)
    polys = polygons or load_polygons()
    if files is None and patterns:
        files = sorted({f for p in patterns for f in list_datasets(p)})
    if files is None:
        files = list_datasets("*"); random.shuffle(files); files = files[:n_fields]
    files = [f for f in files if parse_truth(os.path.basename(f)) is not None]
    if verbose:
        print(f"fields={len(files)}  polygons={list(polys)}  detrend={detrend}")

    recs = []
    for k, path in enumerate(files):
        fam, rng_, sill = parse_truth(os.path.basename(path))
        tm = true_model(fam, rng_, sill)
        for calibrate in (False, True):
            try:
                est = run_one(path, next(iter(polys.values())), calibrate, detrend=detrend,
                              max_samples=max_samples, n_boot=n_boot)
            except Exception as e:
                if verbose:
                    print(f"  [{k}] {os.path.basename(path)} calibrate={calibrate} FAILED: {type(e).__name__}: {e}")
                continue
            for pname, poly in polys.items():
                # Re-target the estimator to THIS polygon (previously the
                # estimator stayed bound to the first polygon while the
                # truth used the loop polygon, corrupting coverage rows).
                est.polygon = poly
                est.area = float(poly.area)
                est.calc_mean_correlated_polygon(n_pairs=N_PAIRS, seed=seed)
                lo, hi, ctr = (est.mean_correlated_polygon_min, est.mean_correlated_polygon_max,
                               est.mean_correlated_polygon)
                # "truth": analytic sigma_A of the PRESCRIBED model over the polygon.
                # Default uses the workflow's own MC integrator; pass truth_fn(family,
                # range, sill, poly, path) to supply an independent truth (e.g. your
                # notebook's closed-form unc_analytic / unc_fft, or an ensemble std).
                st = (truth_fn(fam, rng_, sill, poly, path) if truth_fn is not None
                      else est.estimate_std_mean_monte_carlo(poly, tm, tm.get_total_sill(),
                                                             n_pairs=N_PAIRS, seed=seed + 1))
                if None in (lo, hi, ctr) or ctr <= 0:
                    continue
                recs.append(dict(file=os.path.basename(path), family=fam, polygon=pname,
                                 mode="calibrated" if calibrate else "legacy",
                                 true_sigmaA=st, lo=lo, hi=hi, center=ctr,
                                 inside=bool(lo <= st <= hi), rel_width=(hi - lo) / max(ctr, 1e-9)))
        if verbose and (k + 1) % 5 == 0:
            print(f"  processed {k+1}/{len(files)}")

    cols = ["file", "family", "polygon", "mode", "true_sigmaA", "lo", "hi",
            "center", "inside", "rel_width"]
    try:
        import pandas as pd
        df = pd.DataFrame(recs, columns=cols)   # keep columns even when empty
    except Exception:
        return recs
    if len(df) == 0:
        print("\n[!] No records produced -- every run_one() call failed or returned None.")
        print("    Most common cause: the installed `topochange` does not include the P0 edits")
        print("    (i.e. it is not an editable install of this repo). Verify with:")
        print("      import topochange, inspect")
        print("      inspect.signature(topochange.SingleVariogram.compute_empirical_variogram)"
              ".parameters.keys()   # expect 'detrend_order'")
        print("      inspect.signature(topochange.RegionalUncertaintyEstimator.__init__)"
              ".parameters.keys()    # expect 'calibrate'")
        print("    If those are missing, reinstall editable from the repo root:  pip install -e .")
        print("    To see the underlying exception directly, run: "
              "validate_p1_coverage.debug_one()")
        return df
    if verbose:
        print("\n================  P0-1 ENVELOPE CALIBRATION  ================")
        for mode in ("legacy", "calibrated"):
            d = df[df["mode"] == mode]
            if len(d):
                print(f"  {mode:11s}: 68% coverage = {d['inside'].mean():.3f}   "
                      f"mean rel. width = {d['rel_width'].mean():.3f}   (n={len(d)})")
        print("  target coverage ~ 0.68 (calibrated should exceed legacy)")
        print("============================================================")
    return df

def debug_one(path=None, calibrate=True, detrend=0):
    """Run a SINGLE field with no error handling so the real traceback surfaces."""
    polys = load_polygons()
    if path is None:
        cand = list_datasets("Spherical_r100_s0-1_n0_*") or list_datasets("*")
        path = cand[0]
    print("debug field :", os.path.basename(path))
    print("polygons    :", list(polys))
    est = run_one(path, next(iter(polys.values())), calibrate, detrend=detrend)
    est.calc_mean_correlated_polygon(n_pairs=N_PAIRS, seed=SEED)
    print("OK -> min=%s  max=%s  center=%s" % (
        est.mean_correlated_polygon_min, est.mean_correlated_polygon_max,
        est.mean_correlated_polygon))
    return est

# ---------------------------------------------------------------------------
# Option 3 -- GOLD STANDARD: per-config coverage from a regenerated realization
# ensemble.  Matches the dataset generator (circulant embedding + gstools) and
# the notebook's unc_fft / unc_simulation conventions.
# ---------------------------------------------------------------------------
_FINITE_RANGE = {"Spherical", "Circular", "Cubic", "HyperSpherical", "SuperSpherical"}
_GS = {"Spherical": "Spherical", "Exponential": "Exponential", "Gaussian": "Gaussian",
       "Matern": "Matern", "Stable": "Stable"}

def _gs_model(family, target_range, structured_sill, nu=1.5, alpha=1.5):
    """gstools CovModel for a prescribed EFFECTIVE range (len_scale convention as
    in 4_Generate_random_fields.ipynb)."""
    import gstools as gs
    cls = getattr(gs, _GS[family])
    extra = {}
    if family == "Matern": extra["nu"] = nu
    if family == "Stable": extra["alpha"] = alpha
    if cls.__name__ in _FINITE_RANGE:
        ls = float(target_range)
    else:
        ls = float(target_range) / cls(dim=2, var=1.0, len_scale=1.0, **extra).percentile_scale(0.95)
    return cls(dim=2, var=float(structured_sill), len_scale=ls, **extra)

def _grf_circulant(model, shape, res=1.0, seed=None, pad=2):
    """Exact stationary GRF via circulant embedding (Dietrich & Newsam 1993) --
    identical to the dataset generator in 4_Generate_random_fields.ipynb."""
    ny, nx = shape; My, Mx = pad * ny, pad * nx
    ky = np.concatenate([np.arange(0, My // 2 + 1), np.arange(My // 2 + 1 - My, 0)])
    kx = np.concatenate([np.arange(0, Mx // 2 + 1), np.arange(Mx // 2 + 1 - Mx, 0)])
    Y, X = np.meshgrid(ky * res, kx * res, indexing="ij")
    lam = np.fft.fft2(model.covariance(np.hypot(X, Y))).real
    lam[lam < 0] = 0.0
    r = np.random.default_rng(seed)
    xi = r.normal(size=(My, Mx)) + 1j * r.normal(size=(My, Mx))
    return (np.fft.fft2(xi * np.sqrt(lam)) / np.sqrt(My * Mx)).real[:ny, :nx]

def _pad_for(rng, dom=1000, res=1.0):
    return min(4, max(2, math.ceil((dom + 3 * rng / res) / dom)))

def _poly_mask(transform, shape, poly):
    import shapely
    H, W = shape
    xs = transform.c + transform.a * (np.arange(W) + 0.5)
    ys = transform.f + transform.e * (np.arange(H) + 0.5)
    gx, gy = np.meshgrid(xs, ys)
    return shapely.contains_xy(poly, gx, gy)

def _sigmaA_analytic(model, mask, res):
    """Exact sigma_A of the TRUE model over a polygon mask via FFT autocorrelation
    (areal-average variance; same as the notebook's unc_fft)."""
    H, W = mask.shape; N = int(mask.sum())
    fy, fx = 2 * H, 2 * W
    F = np.fft.rfft2(mask.astype(float), s=(fy, fx))
    ac = np.fft.irfft2(F * np.conj(F), s=(fy, fx))
    dy = np.fft.fftfreq(fy) * fy; dx = np.fft.fftfreq(fx) * fx
    DY, DX = np.meshgrid(dy, dx, indexing="ij")
    dist = res * np.hypot(DY, DX)
    v = float((ac * model.covariance(dist)).sum() / (N * N))
    return math.sqrt(v) if v > 0 else 0.0

def per_config_coverage(family, target_range, sill, nugget_frac=0.0, n_real=30,
                        polygons=None, ref_path=None, modes=("legacy", "calibrated"),
                        max_samples=700, n_boot=100, detrend=0, pad=None, seed0=0,
                        check_truth_ensemble=False, verbose=True):
    """GOLD STANDARD (option 3): regenerate ``n_real`` realizations of a prescribed
    variogram (circulant embedding + gstools, matching the dataset generator), run
    the full workflow on each, and report the fraction whose reported sigma_A
    envelope contains the EXACT true sigma_A over each polygon (nominal 0.68).

    Truth = analytic areal-average variance of the true model over the polygon mask
    (FFT). Set ``check_truth_ensemble=True`` to also print the empirical ensemble
    std as a cross-check. Compares ``modes`` = ("legacy", "calibrated").

    Cost ~ n_real * len(modes) * (fit + bootstrap); the bootstrap is
    O(max_samples^3), so START SMALL (n_real=20, max_samples=600) and scale up.
    """
    import rasterio, tempfile
    polys = polygons or load_polygons()
    if ref_path is None:
        cand = list_datasets(f"{family}_r{target_range}_*") or list_datasets("*")
        ref_path = cand[0]                 # borrow transform/crs so polygons align
    with rasterio.open(ref_path) as src:
        transform, crs = src.transform, src.crs
        shape = (src.height, src.width)
    res = abs(transform.a)
    if pad is None:
        pad = _pad_for(target_range, dom=max(shape), res=res)
    structured = (1.0 - nugget_frac) * sill
    model = _gs_model(family, target_range, structured)
    masks = {pn: _poly_mask(transform, shape, pg) for pn, pg in polys.items()}
    truth = {pn: _sigmaA_analytic(model, masks[pn], res) for pn in polys}
    if verbose:
        print(f"[gold] {family} range={target_range} sill={sill} nugget_frac={nugget_frac} "
              f"grid={shape} pad={pad}")
        print("       true sigma_A:", {k: round(v, 4) for k, v in truth.items()})
        if check_truth_ensemble:
            ens = {pn: [] for pn in polys}
            for k in range(max(n_real, 100)):
                f = _grf_circulant(model, shape, res, seed=10_000 + seed0 + k, pad=pad)
                for pn in polys: ens[pn].append(float(f[masks[pn]].mean()))
            print("       ensemble sigma_A (cross-check):",
                  {pn: round(float(np.std(ens[pn], ddof=1)), 4) for pn in polys})

    prof = {"driver": "GTiff", "count": 1, "dtype": "float32", "transform": transform,
            "crs": crs, "height": shape[0], "width": shape[1], "nodata": None}
    tmp = os.path.join(tempfile.gettempdir(), f"_gold_{family}_{target_range}.tif")
    recs = []
    for k in range(n_real):
        f = _grf_circulant(model, shape, res, seed=seed0 + k, pad=pad)
        if nugget_frac > 0:
            f = f + np.sqrt(nugget_frac * sill) * np.random.default_rng(99_000 + seed0 + k).normal(size=shape)
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(f.astype("float32"), 1)
        for mode in modes:
            calibrate = (mode == "calibrated")
            try:
                est = run_one(tmp, next(iter(polys.values())), calibrate,
                              detrend=detrend, max_samples=max_samples, n_boot=n_boot)
            except Exception as e:
                if verbose:
                    print(f"  real {k} {mode} FAILED: {type(e).__name__}: {e}")
                continue
            for pn, pg in polys.items():
                est.polygon = pg; est.area = float(pg.area)      # re-target (ensemble is polygon-independent)
                est.calc_mean_correlated_polygon(n_pairs=N_PAIRS, seed=SEED)
                lo, hi, ctr = (est.mean_correlated_polygon_min,
                               est.mean_correlated_polygon_max,
                               est.mean_correlated_polygon)
                if None in (lo, hi, ctr) or ctr <= 0:
                    continue
                recs.append(dict(real=k, mode=mode, polygon=pn, true_sigmaA=truth[pn],
                                 lo=lo, hi=hi, center=ctr, inside=bool(lo <= truth[pn] <= hi),
                                 rel_width=(hi - lo) / max(ctr, 1e-9),
                                 center_over_true=ctr / max(truth[pn], 1e-12)))
        if verbose and (k + 1) % 10 == 0:
            print(f"  {k+1}/{n_real} realizations")
    try:
        os.remove(tmp)
    except OSError:
        pass

    try:
        import pandas as pd
        df = pd.DataFrame(recs, columns=["real", "mode", "polygon", "true_sigmaA", "lo",
                                         "hi", "center", "inside", "rel_width", "center_over_true"])
    except Exception:
        return recs
    if verbose and len(df):
        print("\n==========  GOLD-STANDARD PER-CONFIG COVERAGE  ==========")
        for mode in modes:
            d = df[df["mode"] == mode]
            if len(d):
                print(f"  {mode:11s}: coverage = {d['inside'].mean():.3f}   "
                      f"center/true = {d['center_over_true'].median():.2f}   "
                      f"mean rel. width = {d['rel_width'].mean():.3f}   (n={len(d)})")
        print("  target ~0.68 (calibrated should approach it; legacy under-covers)")
        print("=========================================================")
    return df

if __name__ == "__main__":
    run_coverage(n_fields=int(os.environ.get("N_FIELDS", "24")),
                 detrend=int(os.environ.get("DETREND", "0")))
