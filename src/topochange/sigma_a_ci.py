"""Calibrated confidence intervals for the regional uncertainty ``sigma_A``.

Two constructions (see ``SigmaA_CI_review_and_implementation_guide.md``):

Path A -- :func:`field_bootstrap_sigma_a`
    Full-grid parametric *field* bootstrap. Simulates whole synthetic difference
    rasters from the fitted model (or the Akaike-weighted model ensemble) by
    circulant embedding on the real grid/footprint, re-runs the entire estimation
    chain ``E(.)`` on each, and forms a bias-correcting log-scale interval from the
    spread of the re-estimated ``sigma_A``. Captures all three variance sources
    (parameter, model-selection, finite-domain/ergodic). This is the robust but
    compute-heavy *reference*.

Path B -- :func:`analytic_sigma_a_interval`
    Cheap, per-raster parameter+model propagation. Reuses the existing
    ``calibrated_parameter_ensemble`` (a per-model parametric variogram bootstrap
    on a point subsample) and combines across candidate structures with the
    Buckland et al. (1997) model-averaged standard error. No whole-field
    simulation. Must be certified against Path A before it is trusted (it is
    Gaussian and fixed-structure-per-model).

Shared building blocks: :func:`simulate_field_from_model` (circulant embedding
straight from a fitted ``CompositeVariogramModel``), :func:`log_basic_interval`
(the bias-correcting log interval), :func:`buckland_model_averaged_se`, and
:func:`experimental_variogram_covariance_gaussian` (the Gaussian VARCOV kernel of
Pardo-Iguzquiza & Dowd, 2001, used by the reference analytic route).

References
----------
Dietrich, C. R., & Newsam, G. N. (1997). SIAM J. Sci. Comput., 18(4), 1088-1107.
Buckland, S. T., Burnham, K. P., & Augustin, N. H. (1997). Biometrics, 53(2), 603-618.
Pardo-Iguzquiza, E., & Dowd, P. A. (2001). Math. Geol., 33(4), 397-419.
Marchant, B. P., & Lark, R. M. (2004). Math. Geol., 36(8), 867-898.
"""

from __future__ import annotations

import math
import os
import tempfile
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import norm

__all__ = [
    "simulate_field_from_model",
    "read_template",
    "simulate_from_fitted",
    "log_basic_interval",
    "make_estimate_fn",
    "field_bootstrap_sigma_a",
    "bootstrap_ci",
    "buckland_model_averaged_se",
    "analytic_sigma_a_interval",
    "experimental_variogram_covariance_gaussian",
]

NODATA = -9999.0


# ======================================================================
# Shared: simulate a difference field from a fitted composite model
# ======================================================================

def _correlated_covariance_on_grid(model, ny: int, nx: int, res: float) -> np.ndarray:
    """Correlated-part covariance C_corr(h) on the doubled (torus) grid.

    C_corr(h) = sum_i c_i * rho_i(h) = sum_i [c_i - gamma_i(h)], evaluated on the
    minimum-image separations of a 2*ny by 2*nx grid. The nugget is *excluded*
    here (added as white noise by the caller), matching the circulant-embedding
    convention in ``synthetic_validation/generate.py``.
    """
    my, mx = 2 * ny, 2 * nx
    ix = np.arange(mx)
    iy = np.arange(my)
    dx = np.minimum(ix, mx - ix)[None, :] * res
    dy = np.minimum(iy, my - iy)[:, None] * res
    h = np.hypot(np.broadcast_to(dx, (my, mx)), np.broadcast_to(dy, (my, mx)))

    n_comp = len(model.component_names)
    cov = np.zeros_like(h)
    for i in range(n_comp):
        c_i = float(model.get_component_params(i)[0])          # partial sill
        cov += c_i - model.evaluate_component(i, h)            # c_i * rho_i(h)
    return cov


def simulate_field_from_model(
    model, ny: int, nx: int, res: float, rng: np.random.Generator,
    *, include_nugget: bool = True, t_dof: Optional[float] = None,
) -> np.ndarray:
    """One zero-mean synthetic difference field from a fitted composite ``model``.

    Exact stationary Gaussian simulation by circulant embedding (Dietrich &
    Newsam, 1997), plus the fitted nugget as white noise. This is the same
    algorithm as the harness ``CirculantSimulator`` but driven directly by a
    ``CompositeVariogramModel`` (no ``TrueModel`` adapter needed), so it stays
    inside ``topochange``.

    Parameters
    ----------
    model : CompositeVariogramModel
        Fitted, parameters already set. Must be stationary (bounded components).
    ny, nx : int
        Grid shape (rows, cols).
    res : float
        Pixel size (projected units).
    rng : np.random.Generator
    include_nugget : bool
        Add the fitted nugget as white noise (default True).
    t_dof : float, optional
        If given (>2), rescale the Gaussian to a Student-t marginal with this
        many d.o.f. (variance preserved) to probe heavy tails.
    """
    if not model.is_stationary:
        raise ValueError(
            "simulate_field_from_model requires a stationary (bounded) model; "
            f"unbounded components: {model.unbounded_components}"
        )
    my, mx = 2 * ny, 2 * nx
    cov = _correlated_covariance_on_grid(model, ny, nx, res)
    spec = np.fft.fft2(cov).real
    neg = float(np.abs(spec[spec < 0].sum()) / max(spec[spec > 0].sum(), 1e-300))
    if neg > 1e-3:
        import warnings
        warnings.warn(
            f"Circulant embedding not nonnegative (negative mass fraction {neg:.2e}); "
            f"clipping. Increase grid padding if this grows.", stacklevel=2,
        )
    sqrt_spec = np.sqrt(np.clip(spec, 0.0, None))

    w = rng.standard_normal((my, mx)) + 1j * rng.standard_normal((my, mx))
    f = np.fft.fft2(sqrt_spec * w) / math.sqrt(mx * my)
    field = f.real[:ny, :nx].astype(np.float64)

    if t_dof is not None:
        if t_dof <= 2:
            raise ValueError("t_dof must be > 2 for finite variance")
        field *= math.sqrt((t_dof - 2.0) / t_dof)
        field *= math.sqrt(t_dof / rng.chisquare(t_dof))

    if include_nugget:
        nug = float(model.get_nugget())
        if nug > 0:
            field += rng.standard_normal(field.shape) * math.sqrt(nug)
    return field


def read_template(raster_path: str) -> Dict:
    """Read grid geometry + valid-pixel mask from a template raster.

    Returns a dict with ``ny, nx, res, transform, crs, valid`` (2-D bool mask).
    Note: ``RasterDataHandler.data_array`` is a flattened vector of *finite*
    pixels only, so the 2-D shape/mask must come from the file itself.
    """
    import rasterio
    with rasterio.open(raster_path) as ds:
        band = ds.read(1, masked=True)
        valid = ~np.ma.getmaskarray(band)
        if np.isscalar(valid) or valid.shape != band.shape:
            valid = np.isfinite(np.asarray(band))
        return {
            "ny": int(band.shape[0]), "nx": int(band.shape[1]),
            "res": float(ds.transform.a), "transform": ds.transform,
            "crs": ds.crs, "valid": np.asarray(valid, dtype=bool),
        }


def simulate_from_fitted(model, template: Dict, rng: np.random.Generator,
                         **sim_kwargs) -> np.ndarray:
    """Synthetic difference raster from ``model`` on a template's grid+footprint.

    ``template`` is the dict returned by :func:`read_template`. Invalid pixels
    (holes / outside the footprint) are stamped with ``NODATA``.
    """
    field = simulate_field_from_model(
        model, template["ny"], template["nx"], template["res"], rng, **sim_kwargs
    )
    field[~template["valid"]] = NODATA
    return field


# ======================================================================
# Shared: the bias-correcting log interval
# ======================================================================

def log_basic_interval(
    sigma_hat: float, sigma_star: Sequence[float], alpha: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Bias-correcting *basic* bootstrap interval on the log scale.

    Returns ``((lo, hi), (lo_pct, hi_pct))`` -- the bias-corrected basic interval
    and, for comparison, the (non-bias-corrected) percentile interval. Use
    ``alpha=0.32`` for 68% and ``alpha=0.05`` for 95%.

    The center ``2*log(sigma_hat) - mean(log sigma_star)`` recenters for the
    pipeline's multiplicative bias; ``std(log sigma_star)`` supplies the width.
    Compare the two intervals' *half-width asymmetry* (not their centers) to
    diagnose skew -- the centers differ by ~2x the bias by construction.
    """
    s = np.asarray(sigma_star, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size < 2:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    L = np.log(s)
    m_star, s_star = float(L.mean()), float(L.std(ddof=1))
    z = float(norm.ppf(1 - alpha / 2))
    log_hat = math.log(sigma_hat)
    lo = math.exp(2 * log_hat - m_star - z * s_star)
    hi = math.exp(2 * log_hat - m_star + z * s_star)
    lo_p, hi_p = np.exp(np.percentile(L, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    return (lo, hi), (float(lo_p), float(hi_p))


def _interval_dict(sigma_hat: float, sigma_star: np.ndarray) -> Dict:
    (lo68, hi68), (p68l, p68h) = log_basic_interval(sigma_hat, sigma_star, 0.32)
    (lo95, hi95), (p95l, p95h) = log_basic_interval(sigma_hat, sigma_star, 0.05)
    return {
        "sigma_a": float(sigma_hat),
        "ci68": (lo68, hi68), "ci95": (lo95, hi95),
        "ci68_percentile": (p68l, p68h), "ci95_percentile": (p95l, p95h),
        "n_boot": int(np.sum(np.isfinite(sigma_star))),
        "samples": np.asarray(sigma_star, dtype=float),
    }


# ======================================================================
# Path A: full-grid ensemble field bootstrap  (the reference)
# ======================================================================

def _draw_model_from_ensemble(ensemble, rng: np.random.Generator):
    """Pick a structure by Akaike weight and a parameter draw within it.

    ``ensemble`` = list of ``(composite_model, weight, param_samples)`` exactly as
    produced by ``SingleVariogram.calibrated_parameter_ensemble`` /
    ``FittedVariogramModel.model_ensemble``. Returns a model with parameters set.
    """
    import copy
    if not ensemble:
        raise ValueError("Cannot draw a model from an empty ensemble.")
    weights = np.array([w for _, w, _ in ensemble], dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    total = weights.sum()
    # Degenerate (all-zero / non-finite) weights: fall back to uniform rather
    # than producing NaN probabilities in rng.choice.
    weights = (weights / total if total > 0
               else np.full(len(ensemble), 1.0 / len(ensemble)))
    k = int(rng.choice(len(ensemble), p=weights))
    model, _, samples = ensemble[k]
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    m = copy.deepcopy(model)
    m.set_params(samples[rng.integers(len(samples))])
    return m


def make_estimate_fn(
    *, run_kwargs: Dict, aoi, transform, crs, unit: str = "m",
    resolution: Optional[float] = None, n_pairs: int = 200_000,
    variogram: str = "single", n_realizations: int = 30,
    calibrate: bool = False, ci_seed: Optional[int] = 0,
) -> Callable[[np.ndarray], Dict]:
    """Build the estimation chain ``E(.)`` as a single callable.

    Returns ``estimate_fn(field_array) -> {'sigma_a', 'sigma_tot', 'model'}``. It
    writes the field to a temp GeoTIFF, loads it via ``RasterDataHandler``,
    removes the median, fits a variogram (AICc selection), and propagates
    ``RegionalUncertaintyEstimator`` over ``aoi``. The valid-pixel mask of the
    simulated field (holes / stable-area geometry) is inherited from the template
    it was drawn on, so stable-area masking needs no special handling here.

    Parameters
    ----------
    variogram : {'single', 'grid'}
        ``'single'`` (default, fast): one ``SingleVariogram`` per field -- a
        single empirical variogram + AICc model selection. Drops the
        ``n_realizations`` pixel-resampling (the small within-field "sampling"
        term), so ~``n_realizations``x cheaper; this is the minutes-long path.
        ``'grid'``: ``GridVariogram`` with ``n_realizations`` resamples, matching
        a GridVariogram-reported point estimate exactly, ~``n_realizations``x
        slower.
    run_kwargs : dict
        The variogram config: area_side, samples_per_area, max_samples, bin_width,
        max_lag_multiplier, estimator (empirical-variogram stage) and model_types,
        max_components, criterion (model-fit stage). ``seed`` is injected per call.
    aoi, transform, crs, n_pairs, calibrate : see the module guide.
    """
    import rasterio
    from .variogram import GridVariogram, SingleVariogram, RasterDataHandler
    from .uncertainty import RegionalUncertaintyEstimator

    res = float(resolution if resolution is not None else transform.a)
    # ``seed`` must not ride along inside run_kwargs: the grid path calls
    # ``gv.run(seed=ci_seed, **run_kwargs)`` and a duplicate would raise
    # TypeError inside every replicate (after the fields are paid for).
    run_kwargs = {k: v for k, v in dict(run_kwargs).items() if k != "seed"}
    _emp = ("area_side", "samples_per_area", "max_samples", "bin_width",
            "max_lag_multiplier", "estimator", "detrend_order")
    _fit = ("model_types", "max_components", "criterion", "include_nugget")
    emp_kwargs = {k: run_kwargs[k] for k in _emp if k in run_kwargs}
    fit_kwargs = {k: run_kwargs[k] for k in _fit if k in run_kwargs}
    fit_kwargs.setdefault("include_nugget", True)

    def _write(path, field):
        with rasterio.open(
            path, "w", driver="GTiff", height=field.shape[0], width=field.shape[1],
            count=1, dtype="float64", crs=crs, transform=transform, nodata=NODATA,
        ) as dst:
            dst.write(field.astype("float64"), 1)

    def _fit_variogram(rdh_br):
        if variogram == "grid":
            gv = GridVariogram(rdh_br, n_realizations=n_realizations)
            gv.run(seed=ci_seed, **run_kwargs)
            return gv
        sv = SingleVariogram(rdh_br)
        sv.compute_empirical_variogram(seed=ci_seed, **emp_kwargs)
        sv.fit_model(seed=ci_seed, **fit_kwargs)
        return sv

    def estimate_fn(field: np.ndarray) -> Dict:
        tmpdir = tempfile.mkdtemp(prefix="sigma_a_ci_")
        p_raw = os.path.join(tmpdir, "sim.tif")
        p_br = os.path.join(tmpdir, "sim_biasremoved.tif")
        try:
            _write(p_raw, field)
            rdh = RasterDataHandler(p_raw, unit, res)
            rdh.load_raster()
            bias = float(np.median(rdh.data_array))
            rdh.subtract_value_from_raster(p_br, bias)
            rdh_br = RasterDataHandler(p_br, unit, res)
            rdh_br.load_raster()

            va = _fit_variogram(rdh_br)
            est = RegionalUncertaintyEstimator(
                raster_data_handler=rdh_br, variogram_analysis=va,
                area_of_interest=aoi, fitted_model=va.fitted_model,
                calibrate=calibrate,
            )
            est.calc_total_uncertainty(n_pairs=n_pairs, seed=ci_seed)
            try:
                model = va.fitted_model.composite_model.structural_description()
            except Exception:
                model = None
            return {
                "sigma_a": est.mean_correlated_polygon,
                "sigma_tot": est.total_uncertainty_polygon,
                "sigma_a_raster": est.mean_correlated_raster,
                "sigma_tot_raster": est.total_uncertainty_raster,
                "model": model,
            }
        except Exception as e:  # a failed replicate must not kill the run
            return {"sigma_a": float("nan"), "sigma_tot": float("nan"),
                    "sigma_a_raster": float("nan"),
                    "sigma_tot_raster": float("nan"),
                    "model": None, "error": f"{type(e).__name__}: {e}"}
        finally:
            for p in (p_raw, p_br):
                try:
                    os.remove(p)
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

    return estimate_fn


def field_bootstrap_sigma_a(
    source, template: Dict, estimate_fn: Callable[[np.ndarray], Dict],
    sigma_a_hat: float, *, sigma_tot_hat: Optional[float] = None,
    sigma_a_raster_hat: Optional[float] = None,
    sigma_tot_raster_hat: Optional[float] = None,
    B: int = 199, seed: int = 0, n_jobs: int = 1,
    t_dof: Optional[float] = None,
) -> Dict:
    """Path A -- ensemble field bootstrap of the whole sigma_A estimator.

    Parameters
    ----------
    source : ensemble list OR a single CompositeVariogramModel
        If a list of ``(model, weight, param_samples)`` (an Akaike-weighted
        ensemble), each replicate draws a structure by weight and a parameter
        draw within it, so *simulation* model-uncertainty enters the fields. If a
        single fitted model, all fields are drawn from it (selection uncertainty
        then enters only via AICc re-selection inside ``estimate_fn``).
    template : dict
        From :func:`read_template` (grid + footprint of the real raster).
    estimate_fn : callable
        The production chain, from :func:`make_estimate_fn`.
    sigma_a_hat, sigma_tot_hat : float
        The production point estimates the interval is reported around.
    B : int
        Number of synthetic fields (bootstrap replicates).
    n_jobs : int
        If > 1, uses ``joblib`` (falls back to serial if unavailable).

    Returns
    -------
    dict with ``sigma_a`` results (``ci68``, ``ci95``, percentile companions,
    ``samples``) and, if ``sigma_tot_hat`` is given, a parallel ``sigma_tot``
    block.
    """
    is_ensemble = isinstance(source, (list, tuple)) and len(source) > 0 \
        and isinstance(source[0], (list, tuple))

    def one(b: int) -> Dict:
        rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(b,)))
        model = _draw_model_from_ensemble(source, rng) if is_ensemble else source
        field = simulate_from_fitted(model, template, rng, t_dof=t_dof)
        return estimate_fn(field)

    if n_jobs and n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            out = Parallel(n_jobs=n_jobs)(delayed(one)(b) for b in range(B))
        except Exception:
            out = [one(b) for b in range(B)]
    else:
        out = [one(b) for b in range(B)]

    sa = np.array([r.get("sigma_a", np.nan) for r in out], dtype=float)
    result = _interval_dict(sigma_a_hat, sa)
    result["n_failed"] = int(np.sum(~np.isfinite(sa)))
    if sigma_tot_hat is not None:
        st = np.array([r.get("sigma_tot", np.nan) for r in out], dtype=float)
        result["sigma_tot"] = _interval_dict(sigma_tot_hat, st)
    # raster-scale (footprint) companions, when point estimates are supplied
    if sigma_a_raster_hat is not None:
        sar = np.array([r.get("sigma_a_raster", np.nan) for r in out],
                       dtype=float)
        if np.sum(np.isfinite(sar) & (sar > 0)) >= 2:
            result["sigma_a_raster"] = _interval_dict(sigma_a_raster_hat, sar)
    if sigma_tot_raster_hat is not None:
        stt = np.array([r.get("sigma_tot_raster", np.nan) for r in out],
                       dtype=float)
        if np.sum(np.isfinite(stt) & (stt > 0)) >= 2:
            result["sigma_tot_raster"] = _interval_dict(
                sigma_tot_raster_hat, stt)
    # cross-field model-selection proportions (finite-domain model uncertainty)
    models = [r.get("model") for r in out if r.get("model")]
    if models:
        from collections import Counter
        c = Counter(models); tot = sum(c.values())
        result["model_proportions"] = {k: v / tot for k, v in c.most_common()}
    return result


def bootstrap_ci(
    variogram_analysis, aoi, run_kwargs, *, template_raster_path=None, rdh=None,
    B: int = 100, replicate: str = "match", source: str = "ensemble",
    n_pairs: int = 25_000, n_jobs: int = 1, seed: int = 0,
) -> Dict:
    """Per-raster sigma_A confidence interval by field bootstrap.

    Given a variogram already fit to the real raster (``variogram_analysis`` --
    a ``GridVariogram`` or ``SingleVariogram``), this:

    1. reads the production point estimate sigma_A / sigma_tot over ``aoi``;
    2. simulates ``B`` fields from the fitted model, or from the stored model
       ensemble (``source='ensemble'``; for a ``GridVariogram`` this
       is the equal-weighted per-realization ensemble, not an Akaike-weighted
       candidate ensemble), on the raster's grid + valid mask;
    3. refits each field with the SAME estimator type as the point estimate
       (``replicate='match'``, default): ``GridVariogram`` replicates for a
       GridVariogram point estimate, ``SingleVariogram`` for a SingleVariogram.
       The bootstrap bias correction and interval width are only calibrated
       when the replicate estimator matches the point estimator; pass
       ``replicate='single'`` explicitly for the fast-but-uncertified path
       (a warning is emitted when the estimators differ);
    4. forms the bias-correcting log-scale 68/95% interval.

    Returns the :func:`field_bootstrap_sigma_a` dict (``ci68``, ``ci95``,
    ``samples``, ``model_proportions`` -- the cross-field model-selection
    frequencies, ...) plus ``point_estimate``.
    """
    from .uncertainty import RegionalUncertaintyEstimator

    is_grid = hasattr(variogram_analysis, "n_realizations") or (
        type(variogram_analysis).__name__ == "GridVariogram")
    if replicate == "match":
        replicate = "grid" if is_grid else "single"
    elif (replicate == "single") == is_grid:  # explicit mismatch
        warnings.warn(
            "bootstrap_ci: the point estimate comes from a "
            f"{'GridVariogram' if is_grid else 'SingleVariogram'} but "
            f"replicates use replicate='{replicate}'. Mixing estimators "
            "breaks the bootstrap bias correction and the interval is "
            "uncertified; use replicate='match' unless you are deliberately "
            "trading calibration for speed.",
            UserWarning, stacklevel=2,
        )

    rdh = rdh if rdh is not None else getattr(variogram_analysis, "rdh", None)
    if rdh is None:
        raise ValueError("Provide rdh= or a variogram_analysis that stores one "
                         "(e.g. GridVariogram.rdh).")
    path = template_raster_path or getattr(rdh, "raster_path", None)
    if path is None:
        raise ValueError("Provide template_raster_path= (the raster's file path).")

    # 1. production point estimate from the already-fitted model (build ensemble
    #    here if we'll draw simulation structures from it).
    est = RegionalUncertaintyEstimator(
        raster_data_handler=rdh, variogram_analysis=variogram_analysis,
        area_of_interest=aoi, fitted_model=variogram_analysis.fitted_model,
        calibrate=(source == "ensemble"),
    )
    est.calc_total_uncertainty(n_pairs=n_pairs, seed=seed)
    sa_hat, st_hat = est.mean_correlated_polygon, est.total_uncertainty_polygon
    sa_raster_hat = est.mean_correlated_raster
    st_raster_hat = est.total_uncertainty_raster

    # 2. bootstrap source: Akaike-weighted ensemble, or the single central model
    fm = variogram_analysis.fitted_model
    ens = getattr(fm, "model_ensemble", None)
    src = ens if (source == "ensemble" and ens) else fm.composite_model

    # 3. grid geometry + valid mask from the raster; refit each simulated field
    tmpl = read_template(path)
    est_fn = make_estimate_fn(
        run_kwargs=run_kwargs, aoi=aoi, transform=tmpl["transform"], crs=tmpl["crs"],
        unit=getattr(rdh, "unit", "m"), resolution=tmpl["res"], n_pairs=n_pairs,
        variogram=replicate,
        n_realizations=getattr(variogram_analysis, "n_realizations", 30),
        calibrate=False, ci_seed=seed,
    )
    res = field_bootstrap_sigma_a(
        src, tmpl, est_fn, sigma_a_hat=sa_hat, sigma_tot_hat=st_hat,
        sigma_a_raster_hat=sa_raster_hat, sigma_tot_raster_hat=st_raster_hat,
        B=B, seed=seed, n_jobs=n_jobs,
    )
    res["point_estimate"] = {
        "sigma_a": sa_hat, "sigma_tot": st_hat,
        "sigma_a_raster": sa_raster_hat, "sigma_tot_raster": st_raster_hat,
    }
    return res


# ======================================================================
# Path B: analytic parameter + model propagation  (cheap, per-raster)
# ======================================================================

def buckland_model_averaged_se(
    point_estimates: Sequence[float], within_sd: Sequence[float],
    weights: Sequence[float],
) -> Tuple[float, float]:
    """Model-averaged estimate and unconditional SE (Buckland et al., 1997).

    ``s_bar = sum_m w_m * s_m`` and
    ``SE = sum_m w_m * sqrt( within_sd_m^2 + (s_m - s_bar)^2 )`` -- combining
    within-model parameter variance and between-model dispersion. Weights are
    normalized internally.
    """
    s = np.asarray(point_estimates, float)
    v = np.asarray(within_sd, float) ** 2
    w = np.asarray(weights, float)
    w = w / w.sum()
    s_bar = float(np.sum(w * s))
    se = float(np.sum(w * np.sqrt(v + (s - s_bar) ** 2)))
    return s_bar, se


def _sigma_a_pairs(aoi, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Pairwise distances between uniform random point pairs in ``aoi``."""
    import shapely
    minx, miny, maxx, maxy = aoi.bounds
    need = n_pairs * 2
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    got = 0
    while got < need:
        k = int((need - got) * 1.6) + 500
        rx = rng.uniform(minx, maxx, k)
        ry = rng.uniform(miny, maxy, k)
        keep = shapely.contains_xy(aoi, rx, ry)
        xs.append(rx[keep]); ys.append(ry[keep])
        got += int(keep.sum())
    px = np.concatenate(xs)[:need]
    py = np.concatenate(ys)[:need]
    P = np.column_stack([px, py])
    X, Y = P[:n_pairs], P[n_pairs:2 * n_pairs]
    return np.linalg.norm(X - Y, axis=1)


def _sigma_a_of_params(model, params: np.ndarray, h_pairs: np.ndarray) -> float:
    """sigma_A(theta) = sqrt(mean covariance over point pairs in A).

    The caller's ``model`` parameter state is snapshotted and restored, so this
    evaluation never leaves the fitted model / ensemble structure mutated; an
    analytic-CI call must not perturb the object the plot and other downstream
    readers rely on.
    """
    _saved = model.params
    _saved = None if _saved is None else np.asarray(_saved).copy()
    try:
        model.set_params(np.asarray(params, dtype=float))
        s2 = model.get_total_sill()
        if s2 is None or s2 <= 0:
            return float("nan")
        cov = s2 - model(h_pairs)
        return math.sqrt(max(float(np.mean(cov)), 0.0))
    finally:
        model._params = _saved


def analytic_sigma_a_interval(
    variogram_analysis, aoi, *, n_boot: int = 300, n_pairs: int = 100_000,
    seed: Optional[int] = 0, add_model_uncertainty: bool = True,
    within_source: str = "parametric",
) -> Dict:
    """Path B -- cheap parameter+model propagation, per raster.

    Two variance sources are combined into a model-averaged SE (Buckland et al.,
    1997) and turned into a log-scale interval:

    * **within-model** -- the finite-domain ("source 3") spread of ``sigma_A``.
      With ``within_source='parametric'`` (default) it comes from
      ``variogram_analysis.parametric_bootstrap_parameters`` -- the *model-based*
      variogram bootstrap that simulates Gaussian samples on the raster geometry
      and refits. This is the engine that actually captures the finite-domain
      fluctuation. With ``within_source='ensemble'`` it uses the object's stored
      ``model_ensemble`` samples instead -- for a ``GridVariogram`` those are
      per-*realization* fits (within-field resampling only), which UNDER-DISPERSE
      and reproduce the current bands' under-coverage. Use 'ensemble' only to
      reproduce that baseline.

    * **between-model** -- structure-selection spread, from the per-structure
      point estimates weighted by their Akaike/selection weights
      (``fitted_model.model_ensemble``).

    No whole-field simulation; seconds-to-minutes per raster. **Certify against
    Path A before trusting** -- Gaussian, and its within-model dispersion inherits
    the point-subsample fidelity of ``parametric_bootstrap_parameters`` (raise its
    ``max_cov_points`` / ``n_realizations`` if it under-covers vs Path A).

    Returns the model-averaged point estimate, SE, log-scale 68/95% intervals,
    the within/between split, and the per-structure breakdown.
    """
    rng = np.random.default_rng(seed)
    h_pairs = _sigma_a_pairs(aoi, n_pairs, rng)
    central = variogram_analysis.fitted_model.composite_model
    central_params = np.asarray(central.get_params() if hasattr(central, "get_params")
                                else variogram_analysis.fitted_model.params, float)
    s_hat = _sigma_a_of_params(central, central_params, h_pairs)

    # -- within-model (finite-domain) spread of sigma_A ---------------------
    if within_source == "parametric":
        samp = variogram_analysis.parametric_bootstrap_parameters(
            n_realizations=n_boot, seed=seed,
        )
        samp = np.atleast_2d(np.asarray(samp, float))
        draws = np.array([_sigma_a_of_params(central, p, h_pairs) for p in samp])
        draws = draws[np.isfinite(draws)]
        within_var = float(np.var(draws, ddof=1)) if draws.size > 1 else 0.0
        within_samples = draws
    else:  # 'ensemble' -- the under-dispersed baseline, for comparison only
        ens = variogram_analysis.fitted_model.model_ensemble or []
        pooled = []
        for model, weight, samples in ens:
            for p in np.atleast_2d(np.asarray(samples, float)):
                v = _sigma_a_of_params(model, p, h_pairs)
                if np.isfinite(v):
                    pooled.append(v)
        within_var = float(np.var(pooled, ddof=1)) if len(pooled) > 1 else 0.0
        within_samples = np.asarray(pooled, float)

    # -- between-model (structure-selection) spread -------------------------
    breakdown, s_list, w_list = [], [], []
    ens = variogram_analysis.fitted_model.model_ensemble if add_model_uncertainty else None
    if ens:
        for model, weight, samples in ens:
            sm = _sigma_a_of_params(
                model, np.atleast_2d(np.asarray(samples, float)).mean(axis=0), h_pairs)
            if not np.isfinite(sm):
                continue
            s_list.append(sm); w_list.append(float(weight))
            breakdown.append({"model": getattr(model, "structural_description",
                                                lambda: str(model))(),
                              "weight": float(weight), "sigma_a": sm})
    if not s_list:                                   # single-structure fallback
        s_list, w_list = [s_hat], [1.0]
        breakdown = [{"model": "central", "weight": 1.0, "sigma_a": s_hat}]

    # Buckland unconditional SE: same within-model variance for each structure
    # (approximation; per-structure parametric bootstrap would refine it).
    s = np.asarray(s_list, float); w = np.asarray(w_list, float); w = w / w.sum()
    s_bar = float(np.sum(w * s))
    within_sd = math.sqrt(within_var)
    se = float(np.sum(w * np.sqrt(within_var + (s - s_bar) ** 2)))

    z68, z95 = float(norm.ppf(0.84)), float(norm.ppf(0.975))
    rel = se / s_bar if s_bar > 0 else float("nan")
    return {
        "sigma_a": s_bar, "se": se,
        "within_sd": within_sd,
        "between_sd": float(np.sqrt(np.sum(w * (s - s_bar) ** 2))),
        "ci68": (s_bar * math.exp(-z68 * rel), s_bar * math.exp(z68 * rel)),
        "ci95": (s_bar * math.exp(-z95 * rel), s_bar * math.exp(z95 * rel)),
        "per_model": breakdown,
        "within_samples": within_samples,
    }


# ======================================================================
# Reference analytic building block: Gaussian VARCOV of the experimental
# variogram (Pardo-Iguzquiza & Dowd, 2001). Unit-tested against Monte Carlo.
# ======================================================================

def experimental_variogram_covariance_gaussian(
    coords: np.ndarray, gamma_func: Callable[[np.ndarray], np.ndarray],
    bin_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Covariance matrix of the Matheron experimental variogram, Gaussian case.

    For a zero-mean Gaussian field with (isotropic) variogram ``gamma_func``, the
    covariance between binned estimates ``ghat_k`` and ``ghat_l`` is

        Cov(ghat_k, ghat_l)
            = 1/(2 N_k N_l) * sum_{(i,j) in P_k} sum_{(u,v) in P_l} G_{ij,uv}^2,
        G_{ij,uv} = gamma(x_i - x_v) + gamma(x_j - x_u)
                    - gamma(x_i - x_u) - gamma(x_j - x_v),

    which follows from ``Cov(D^2, E^2) = 2 Cov(D, E)^2`` for jointly Gaussian
    increments ``D = Z_i - Z_j`` (Pardo-Iguzquiza & Dowd, 2001). This is the
    finite-domain sampling covariance of the variogram -- the ingredient that
    carries "variance source 3" into an analytic cov(theta_hat).

    Cost is O(P^2) in the number of pairs, so pass a **subsample** of points
    (a few tens to a couple hundred). Returns ``(Sigma, ghat_expected, lag_mid)``
    where ``ghat_expected[k] = gamma(lag_mid_k)`` is the model's expected bin
    value and ``Sigma`` is the covariance of the estimates.

    This function is a validated building block for the reference analytic route;
    wiring it through the WLS Jacobian to cov(theta_hat) (Marchant & Lark, 2004)
    and through the Krige integral is left to the caller (see the guide) and
    should be certified against Path A.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    iu, ju = np.triu_indices(n, k=1)
    dvec = np.linalg.norm(coords[iu] - coords[ju], axis=1)
    bin_edges = np.asarray(bin_edges, dtype=float)
    which = np.digitize(dvec, bin_edges) - 1
    nb = len(bin_edges) - 1

    # full gamma matrix on the subsample (n x n), used for G_{ij,uv}
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    Gamma = gamma_func(D)

    bins = [np.where(which == k)[0] for k in range(nb)]
    lag_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ghat_expected = gamma_func(lag_mid)

    Sigma = np.full((nb, nb), np.nan)
    for k in range(nb):
        Pk = bins[k]
        if Pk.size == 0:
            continue
        ik, jk = iu[Pk], ju[Pk]
        for l in range(k, nb):
            Pl = bins[l]
            if Pl.size == 0:
                continue
            il, jl = iu[Pl], ju[Pl]
            # G[a,b] for a in Pk, b in Pl
            G = (Gamma[np.ix_(ik, jl)] + Gamma[np.ix_(jk, il)]
                 - Gamma[np.ix_(ik, il)] - Gamma[np.ix_(jk, jl)])
            val = np.sum(G ** 2) / (2.0 * Pk.size * Pl.size)
            Sigma[k, l] = Sigma[l, k] = val
    return Sigma, ghat_expected, lag_mid
