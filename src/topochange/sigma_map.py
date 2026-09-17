"""Per-pixel error maps: binned σ estimation, export, and validation.

This module complements :mod:`topochange.heteroscedastic` with the pieces a
*per-pixel uncertainty map* workflow needs but which the GAM path does not
provide:

1. :class:`BinnedSigmaModel` / :func:`fit_binned_sigma_model`: an
   N-dimensional **binned-NMAD interpolant** for σ(x).  This is the estimator
   actually used by ``xdem.spatialstats.interp_nd_binning`` /
   ``infer_heteroscedasticity_from_stable``, i.e. the reference implementation
   of Hugonnet et al. (2022), who write that they "numerically model the
   empirical dispersion as a function of the terrain- and sensor-dependent
   variables by multidimensional linear interpolation of the binned data".
   :func:`~topochange.heteroscedastic.fit_sigma_model` instead smooths the same
   bin table with a GAM.  Having both, behind one ``.predict(df)`` interface,
   lets a workflow choose between them on held-out stable terrain
   (:func:`cross_validate_sigma_models`) rather than by assertion.

2. :func:`write_sigma_geotiff`: georeferenced export of the σ field, which the
   heteroscedastic pipeline computes but never writes.

3. Validation that is not circular.  ``NMAD(z) = 1`` is *enforced* by the
   second step of :func:`~topochange.heteroscedastic.standardize`
   (Hugonnet's two-step standardisation), so a global unit NMAD says nothing
   about whether the heteroscedastic *shape* is right.  The honest checks are
   :func:`calibration_by_bin` (does NMAD(z) stay ≈ 1 *within* predictor bins?),
   :func:`cross_validate_sigma_models` (does it hold on spatially held-out
   terrain?), and :func:`patch_validation` (does the σ field plus correlation
   structure reproduce the observed dispersion of area means?).

Everything here is additive: no existing ``topochange`` behaviour changes, and
:class:`BinnedSigmaModel` is duck-type compatible with
:class:`~topochange.heteroscedastic.HeteroscedasticSigmaModel`, so it can be
passed straight to :func:`~topochange.heteroscedastic.standardize` and
:func:`~topochange.heteroscedastic.evaluate_sigma_raster`.

References
----------
Hugonnet, R., Brun, F., Berthier, E., Dehecq, A., Mannerfelt, E. S., Eckert, N.,
    & Farinotti, D. (2022). Uncertainty analysis of digital elevation models by
    spatial inference from stable terrain. *IEEE JSTARS*, 15, 6456–6472.
    https://doi.org/10.1109/JSTARS.2022.3188922
Rolstad, C., Haug, T., & Denby, B. (2009). Spatially integrated geodetic glacier
    mass balance and its uncertainty based on geostatistical analysis.
    *Journal of Glaciology*, 55(192), 666–680.
    https://doi.org/10.3189/002214309789470950
Schaffrath, K. R., Belmont, P., & Wheaton, J. M. (2015). Landscape-scale
    geomorphic change detection: quantifying spatially variable uncertainty and
    circumventing legacy data issues. *Geomorphology*, 250, 334–348.
    https://doi.org/10.1016/j.geomorph.2015.09.020
Bui, L. K., & Glennie, C. L. (2023). Estimation of lidar-based gridded DEM
    uncertainty with varying terrain roughness and point density. *ISPRS Open
    Journal of Photogrammetry and Remote Sensing*, 7, 100028.
    https://doi.org/10.1016/j.ophoto.2022.100028
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .heteroscedastic import nmad, robust_center

__all__ = [
    "nd_binning",
    "mixture_nmad",
    "BinnedSigmaModel",
    "fit_binned_sigma_model",
    "BinnedBiasModel",
    "fit_binned_bias_model",
    "misregistration_diagnostic",
    "CallableSigmaModel",
    "uniform_edges",
    "trimmed_nmad_scale",
    "write_sigma_geotiff",
    "calibration_by_bin",
    "spatial_block_folds",
    "cross_validate_sigma_models",
    "patch_validation",
    "plot_1d_binning",
    "plot_2d_binning",
    "plot_sigma_map",
]


# ---------------------------------------------------------------------------
# N-dimensional binning
# ---------------------------------------------------------------------------

def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges for ``values``, de-duplicated and widened at the ends.

    Quantile (rather than uniform) edges keep the bin counts balanced for the
    strongly skewed predictors typical here: CHM, point density and roughness
    are all long-tailed, and uniform edges put ~90% of the pixels in the first
    bin.  Duplicate quantiles (a spiked distribution, e.g. a CHM that is exactly
    0 over most of the stable terrain) collapse to fewer bins rather than
    raising.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("No finite values to bin.")
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if edges.size < 3:
        # Degenerate / near-constant predictor: a single bin spanning the range.
        span = max(float(np.ptp(v)), 1e-9)
        edges = np.array([v.min() - 1e-9 * span, v.max() + 1e-9 * span])
    # Nudge the outer edges so the extreme observations are strictly inside.
    edges = edges.astype(float)
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def nd_binning(
    df: pd.DataFrame,
    predictors: Sequence[str],
    *,
    value_col: str = "dh",
    statistic: Callable[[np.ndarray], float] = nmad,
    n_bins: int = 8,
    min_count: int = 100,
    fac_spread_outliers: Optional[float] = 7.0,
    outlier_scope: str = "local",
    edges: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Bin ``value_col`` on a quantile grid of ``predictors`` and reduce each cell.

    Parameters
    ----------
    df : pandas.DataFrame
        Predictor frame (typically stable terrain only), e.g. the output of
        :func:`topochange.heteroscedastic.build_predictor_frame`.
    predictors : sequence of str
        Continuous predictor columns defining the binning axes.
    value_col : str
        Column reduced within each cell (``'dh'`` for a σ model).
    statistic : callable
        Reduction applied to the cell's values.  Default
        :func:`~topochange.heteroscedastic.nmad`, the robust dispersion used by
        Hugonnet et al. (2022) and xDEM.
    n_bins : int
        Quantile bins per predictor.  The joint grid has ``n_bins ** k`` cells,
        so this is the knob that trades resolution against cell occupancy:
        with 5 predictors and ``n_bins=8`` there are 32,768 cells and most will
        be empty.  Three or four predictors is the practical ceiling.
    min_count : int
        Cells with fewer rows are treated as missing (filled by interpolation
        in :func:`fit_binned_sigma_model`).  xDEM's default is 100.
    fac_spread_outliers : float, optional
        Before reducing, drop values further than ``fac_spread_outliers`` times
        a spread estimate from a centre estimate.  Mirrors xDEM's outlier guard
        (default 7).  ``None`` disables it.
    outlier_scope : {'local', 'global'}
        Whether that threshold is computed within each bin (default) or once
        over the whole frame (``'global'``, which is what
        ``xdem.spatialstats.nd_binning`` does).

        The global rule is unsafe precisely when heteroscedasticity is strong,
        which is the case this module exists for.  Measured on the CA
        geometric-distortion scene: the global NMAD of stable-terrain ``dh`` is
        0.102 m, so a 7× global threshold is 0.72 m, but in the steepest slope
        decile the local NMAD is 0.518 m, making that threshold only 1.4× local
        and clipping away the bulk of that bin's genuine spread.  The fitted
        NMAD for the bin collapsed from 0.518 m to 0.236 m, i.e. the guard
        deleted more than half of the very signal the σ model is there to
        capture.  Local scoping removes blunders without flattening the
        heteroscedasticity.
    edges : dict, optional
        Pre-computed ``{predictor: edges}``.  Supply the *training* edges when
        binning a held-out fold so train and test share a grid.

    Returns
    -------
    table : pandas.DataFrame
        One row per occupied cell, with columns ``predictors`` (bin *centres*),
        ``<p>_bin`` (integer bin index per predictor), ``count``, and
        ``statistic`` (the reduced value).  Cells below ``min_count`` are kept
        with ``statistic = NaN`` so callers can see the occupancy pattern.
    edges : dict
        ``{predictor: edges}`` actually used, for reuse on held-out data.
    """
    predictors = list(predictors)
    if not predictors:
        raise ValueError("At least one predictor is required.")
    missing = [p for p in predictors if p not in df.columns]
    if missing:
        raise KeyError(f"Predictors not in frame: {missing}")
    if value_col not in df.columns:
        raise KeyError(f"value_col {value_col!r} not in frame")

    if outlier_scope not in {"local", "global"}:
        raise ValueError("outlier_scope must be 'local' or 'global'")

    work = df[list(predictors) + [value_col]].copy()
    vals = work[value_col].to_numpy(dtype=float)

    if (fac_spread_outliers is not None and outlier_scope == "global"
            and np.isfinite(vals).any()):
        centre = robust_center(vals)
        spread = nmad(vals)
        if np.isfinite(spread) and spread > 0:
            bad = np.abs(vals - centre) > fac_spread_outliers * spread
            vals = np.where(bad, np.nan, vals)
            work[value_col] = vals

    use_edges: Dict[str, np.ndarray] = {}
    for p in predictors:
        if edges is not None and p in edges:
            use_edges[p] = np.asarray(edges[p], dtype=float)
        else:
            use_edges[p] = _quantile_edges(work[p].to_numpy(dtype=float), n_bins)
        # np.digitize with right=False -> index 0 means "below the first edge";
        # subtract 1 so occupied bins run 0 .. n_edges-2, and clip so values at
        # or beyond the outer edges fall in the terminal bins.
        idx = np.digitize(work[p].to_numpy(dtype=float), use_edges[p], right=False) - 1
        idx = np.where(np.isfinite(work[p].to_numpy(dtype=float)), idx, -1)
        idx = np.clip(idx, -1, len(use_edges[p]) - 2)
        work[p + "_bin"] = idx

    key_cols = [p + "_bin" for p in predictors]
    ok = (work[key_cols] >= 0).all(axis=1) & np.isfinite(work[value_col])
    work = work.loc[ok]
    if work.empty:
        raise ValueError("No rows survived binning (all predictors or values NaN).")

    local_clip = fac_spread_outliers is not None and outlier_scope == "local"

    def _reduce(s: pd.Series) -> float:
        v = s.to_numpy(dtype=float)
        if local_clip and v.size >= 10:
            c = robust_center(v)
            sp = nmad(v)
            if np.isfinite(sp) and sp > 0:
                keep = np.abs(v - c) <= fac_spread_outliers * sp
                if keep.sum() >= 5:
                    v = v[keep]
        return float(statistic(v))

    grouped = work.groupby(key_cols, observed=True)[value_col]
    table = grouped.agg(count="size", statistic=_reduce)
    table = table.reset_index()

    # Bin centres, so the interpolant is defined on physical predictor values.
    for p in predictors:
        e = use_edges[p]
        k = table[p + "_bin"].to_numpy(dtype=int)
        table[p] = 0.5 * (e[k] + e[k + 1])

    table.loc[table["count"] < int(min_count), "statistic"] = np.nan
    order = [p + "_bin" for p in predictors] + list(predictors) + ["count", "statistic"]
    return table[order].reset_index(drop=True), use_edges


# ---------------------------------------------------------------------------
# Binned σ model (the xDEM / Hugonnet estimator)
# ---------------------------------------------------------------------------

@dataclass
class BinnedSigmaModel:
    """σ(x) as an N-D interpolant over binned robust dispersion.

    ``σ(x) = interp(continuous predictors) · exp(Σ categorical log-offsets)``.

    Duck-type compatible with
    :class:`~topochange.heteroscedastic.HeteroscedasticSigmaModel`: it exposes
    ``predictors``, ``factor_cols``, ``factor_offsets`` and ``predict(df)``, so
    :func:`~topochange.heteroscedastic.standardize` and
    :func:`~topochange.heteroscedastic.evaluate_sigma_raster` accept it
    unchanged.

    Attributes
    ----------
    predictors : list of str
        Continuous predictors, in interpolation-axis order.
    grid_points : list of ndarray
        Ascending bin centres per predictor (the interpolation grid).
    grid_values : ndarray
        Filled N-D array of σ (or ``log σ`` when ``log_space``) on that grid.
    factor_cols, factor_offsets
        Categorical predictors and their ``{column: {level: log_offset}}``
        tables, the same additive-log-offset convention as the GAM model, so
        the two estimators are directly comparable.
    predictor_ranges : dict
        ``{predictor: (min, max)}`` bin-centre range.  Inputs are clipped to it
        before interpolation, which reproduces xDEM's "extrapolation is fixed to
        nearest neighbour by duplicating edge bins" and prevents the
        catastrophic log-σ blow-up that unconstrained extrapolation produces on
        predictors whose stable-terrain range is much narrower than the scene's
        (a CHM is the standard offender).
    log_space : bool
        Interpolate ``log σ`` rather than σ.  ``True`` (default) treats error
        magnitude as multiplicative and guarantees positivity; ``False``
        reproduces ``xdem.spatialstats.interp_nd_binning`` exactly, which
        interpolates the raw statistic.
    n_filled : int
        How many grid cells were filled by interpolation/extrapolation rather
        than observed.  A large fraction means the joint bin grid is too
        sparse; reduce ``n_bins`` or drop a predictor.
    n_observed : int
        Grid cells populated directly from data.
    """

    predictors: List[str]
    grid_points: List[np.ndarray]
    grid_values: np.ndarray
    factor_cols: List[str] = field(default_factory=list)
    factor_offsets: Dict[str, Dict[str, float]] = field(default_factory=dict)
    predictor_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    log_space: bool = True
    n_filled: int = 0
    n_observed: int = 0
    bin_table: Optional[pd.DataFrame] = None
    edges: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from scipy.interpolate import RegularGridInterpolator  # lazy

        self._interp = RegularGridInterpolator(
            tuple(np.asarray(g, dtype=float) for g in self.grid_points),
            np.asarray(self.grid_values, dtype=float),
            method="linear",
            bounds_error=False,
            fill_value=None,  # never reached: inputs are clipped to the grid
        )

    # -- API parity with HeteroscedasticSigmaModel ---------------------------

    def _clip(self, X: np.ndarray) -> np.ndarray:
        """Clip predictor columns to the bin-centre range (nearest-edge rule)."""
        X = np.array(X, dtype=float, copy=True)
        for j, p in enumerate(self.predictors):
            rng = self.predictor_ranges.get(p)
            if rng is not None and np.isfinite(rng[0]) and np.isfinite(rng[1]):
                X[:, j] = np.clip(X[:, j], rng[0], rng[1])
        return X

    def _apply_factor_offset(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(df))
        for col in self.factor_cols:
            if col not in df.columns:
                continue
            table = self.factor_offsets.get(col, {})
            vals = df[col].astype(str).to_numpy()
            out = out + np.array([table.get(v, 0.0) for v in vals])
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row σ (NaN where any continuous predictor is non-finite).

        The NaN-skip contract matches
        :meth:`topochange.heteroscedastic.HeteroscedasticSigmaModel.predict`:
        lidar predictors are genuinely undefined at pixels with no returns
        (intensity, incidence, GPS time), and those pixels must come out of the
        σ raster as nodata rather than as a silently imputed value.
        """
        missing = [p for p in self.predictors if p not in df.columns]
        if missing:
            raise KeyError(f"Predictors missing from frame: {missing}")
        X = df[self.predictors].to_numpy(dtype=float)
        ok = np.isfinite(X).all(axis=1)
        out = np.full(len(df), np.nan, dtype=float)
        if ok.any():
            vals = self._interp(self._clip(X[ok]))
            out[ok] = vals
        out = out + self._apply_factor_offset(df) if self.log_space else out
        if self.log_space:
            return np.exp(out)
        # Linear space: factor offsets stay multiplicative for comparability.
        scale = np.exp(self._apply_factor_offset(df))
        return out * scale


def _fill_grid(grid: np.ndarray, points: Sequence[np.ndarray]) -> Tuple[np.ndarray, int]:
    """Fill NaN cells of an N-D grid: linear ``griddata`` then nearest.

    This is the same two-stage strategy as ``xdem.spatialstats.interp_nd_binning``
    ("interpolates nodata values of the irregular N-D binning grid with
    scipy.griddata ... then extrapolates nodata values ... with nearest
    neighbour").  Linear alone leaves the convex hull's exterior empty; nearest
    closes it without inventing a trend.
    """
    from scipy.interpolate import griddata  # lazy

    grid = np.array(grid, dtype=float, copy=True)
    nan_mask = ~np.isfinite(grid)
    n_missing = int(nan_mask.sum())
    if n_missing == 0:
        return grid, 0
    if np.isfinite(grid).sum() == 0:
        raise ValueError("Every bin is empty — nothing to interpolate from.")

    mesh = np.meshgrid(*[np.asarray(p, dtype=float) for p in points], indexing="ij")
    coords = np.stack([m.ravel() for m in mesh], axis=1)
    flat = grid.ravel()
    known = np.isfinite(flat)

    if grid.ndim == 1:
        # griddata's 1-D path is finicky; np.interp is exact and cheaper.
        filled = np.interp(coords[:, 0], coords[known, 0], flat[known])
    else:
        filled = flat.copy()
        if known.sum() >= grid.ndim + 1:
            lin = griddata(coords[known], flat[known], coords[~known], method="linear")
            filled[~known] = lin
        still = ~np.isfinite(filled)
        if still.any():
            near = griddata(
                coords[known], flat[known], coords[still], method="nearest"
            )
            filled[still] = near
    return filled.reshape(grid.shape), n_missing


def _factor_log_offsets(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    *,
    value_col: str = "dh",
    min_count: int = 30,
    shrinkage: float = 100.0,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray]:
    """Sequential shrunk log-offsets per categorical level.

    Identical in form to the offsets estimated inside
    :func:`topochange.heteroscedastic.fit_sigma_model` (empirical-Bayes
    shrinkage ``w = n / (n + shrinkage)``, each factor divided out before the
    next is estimated), so the binned and GAM models differ only in how the
    continuous part is estimated.  Returns the offset tables and the residual
    values with the factor effects removed.
    """
    residual = df[value_col].to_numpy(dtype=float).copy()
    tables: Dict[str, Dict[str, float]] = {}
    for col in factor_cols:
        if col not in df.columns:
            continue
        base = nmad(residual)
        table: Dict[str, float] = {}
        levels = df[col].astype(str).to_numpy()
        for lvl in pd.unique(levels):
            sel = levels == lvl
            n = int(sel.sum())
            if n < min_count:
                continue
            lvl_nmad = nmad(residual[sel])
            if lvl_nmad > 0 and base > 0:
                w = n / (n + float(shrinkage))
                table[str(lvl)] = float(w * np.log(lvl_nmad / base))
        tables[col] = table
        scale = np.array([np.exp(table.get(v, 0.0)) for v in levels])
        residual = residual / np.where(scale > 0, scale, 1.0)
    return tables, residual


def fit_binned_sigma_model(
    df: pd.DataFrame,
    predictors: Sequence[str] = ("slope", "roughness"),
    factor_cols: Sequence[str] = (),
    *,
    n_bins: int = 8,
    min_count: int = 100,
    fac_spread_outliers: Optional[float] = 7.0,
    outlier_scope: str = "local",
    log_space: bool = True,
    value_col: str = "dh",
    edges: Optional[Dict[str, np.ndarray]] = None,
) -> BinnedSigmaModel:
    """Fit σ(x) by N-D interpolation of binned NMAD (the xDEM/Hugonnet estimator).

    The continuous part is a piecewise-linear interpolant of the per-cell NMAD
    over a quantile bin grid; empty cells are filled by ``griddata`` (linear,
    then nearest).  Categorical predictors enter as multiplicative offsets
    estimated exactly as in
    :func:`topochange.heteroscedastic.fit_sigma_model`.

    Compared with the GAM path, this makes *no smoothness assumption*: it will
    follow a genuinely sharp σ–predictor relationship (a slope threshold at
    which geometric distortion switches on, a strip boundary) that spline
    smoothing flattens, at the cost of being noisier where cells are thin.
    Which behaviour is preferable is an empirical question;
    :func:`cross_validate_sigma_models` answers it on held-out terrain.

    Parameters
    ----------
    df : pandas.DataFrame
        Stable-terrain predictor frame with ``value_col`` and the predictors.
    predictors : sequence of str
        Continuous predictors.  Keep this to 2–4: the joint grid is
        ``n_bins ** k`` cells and occupancy falls off fast.
    factor_cols : sequence of str
        Categorical predictors (``'dom_class'``, ``'strip_id'``).
    n_bins, min_count, fac_spread_outliers, outlier_scope
        Passed to :func:`nd_binning`.  ``outlier_scope='local'`` (default)
        matters here: the global rule flattens exactly the high-σ bins the
        model is trying to learn (see :func:`nd_binning`).
    log_space : bool
        Interpolate ``log σ`` (default) or raw σ (xDEM-identical).
    value_col : str
        Column holding the elevation difference.
    edges : dict, optional
        Pre-computed bin edges (for held-out evaluation on a fixed grid).

    Returns
    -------
    BinnedSigmaModel
    """
    predictors = [
        p for p in predictors
        if p in df.columns and np.isfinite(pd.to_numeric(df[p], errors="coerce")).any()
    ]
    if not predictors:
        raise ValueError("No usable continuous predictors after NaN screening.")
    factor_cols = [c for c in factor_cols if c in df.columns]

    work = df.reset_index(drop=True).copy()
    offsets, residual = _factor_log_offsets(work, factor_cols, value_col=value_col)
    work[value_col] = residual

    table, use_edges = nd_binning(
        work,
        predictors,
        value_col=value_col,
        statistic=nmad,
        n_bins=n_bins,
        min_count=min_count,
        fac_spread_outliers=fac_spread_outliers,
        outlier_scope=outlier_scope,
        edges=edges,
    )

    # Assemble the dense grid indexed by bin number.
    grid_points: List[np.ndarray] = []
    for p in predictors:
        e = use_edges[p]
        grid_points.append(0.5 * (e[:-1] + e[1:]))
    shape = tuple(len(g) for g in grid_points)
    grid = np.full(shape, np.nan, dtype=float)

    idx = tuple(table[p + "_bin"].to_numpy(dtype=int) for p in predictors)
    stat = table["statistic"].to_numpy(dtype=float)
    good = np.isfinite(stat) & (stat > 0)
    if not good.any():
        raise ValueError(
            "No bin reached min_count with a positive NMAD. Lower min_count, "
            "reduce n_bins, or use fewer predictors."
        )
    grid[tuple(i[good] for i in idx)] = np.log(stat[good]) if log_space else stat[good]
    n_observed = int(good.sum())

    grid, n_filled = _fill_grid(grid, grid_points)
    if not log_space:
        # Nearest/linear fill can undershoot to <= 0 in raw space; a scale
        # parameter must stay positive or standardize() divides by nonsense.
        floor = float(np.nanmin(stat[good])) * 1e-3
        grid = np.maximum(grid, floor)

    frac_filled = n_filled / max(grid.size, 1)
    if frac_filled > 0.5:
        warnings.warn(
            f"{100 * frac_filled:.0f}% of the {grid.size}-cell predictor grid was "
            f"empty and had to be interpolated ({n_observed} cells observed). The "
            f"σ model is mostly extrapolation — reduce n_bins or use fewer "
            f"predictors.",
            stacklevel=2,
        )

    ranges = {p: (float(g.min()), float(g.max())) for p, g in zip(predictors, grid_points)}

    return BinnedSigmaModel(
        predictors=list(predictors),
        grid_points=grid_points,
        grid_values=grid,
        factor_cols=list(factor_cols),
        factor_offsets=offsets,
        predictor_ranges=ranges,
        log_space=bool(log_space),
        n_filled=n_filled,
        n_observed=n_observed,
        bin_table=table,
        edges=use_edges,
    )


# ---------------------------------------------------------------------------
# Predictor-dependent bias
# ---------------------------------------------------------------------------

@dataclass
class BinnedBiasModel:
    """μ(x): a predictor-dependent *median* offset, as an N-D interpolant.

    The heteroscedastic framework of Hugonnet et al. (2022) models the second
    moment and assumes the first has already been dealt with: their Eq. 4
    splits the error into a bias term removed by coregistration and a random
    term, and the σ machinery only ever sees the latter.  In practice a
    difference raster that has only had a *global* median removed can retain a
    strong predictor-dependent offset: horizontal misregistration, scan-geometry
    distortion and vertical-datum error all produce a mean offset that varies
    with terrain, not just a variance that does.

    That residual bias does not stay in the first moment.  Standardising
    ``(dh − global median) / σ(x)`` when the true centre is ``μ(x)`` pushes the
    unmodelled offset straight into the dispersion, so the two-step rescale
    inflates σ everywhere to absorb it.  Measured on the CA geometric-distortion
    scene: the stable-terrain median ``dh`` ramps from −0.108 m on flat ground to
    −0.391 m above 46° slope, and re-centring per slope bin drops the
    global ``NMAD(z)`` from 1.39 to 1.08, i.e. most of the apparent σ
    mis-scaling was unmodelled bias, not unmodelled noise.

    Fitting is deliberately identical to :class:`BinnedSigmaModel` except that
    the reduced statistic is the robust *centre*, the interpolation is in linear
    space, and categorical factors are additive.  Whether to *subtract* μ(x)
    from ``dh`` (a correction to the change measurement) or only to report it
    (a diagnostic) is a scientific decision the caller must make; see
    :func:`fit_binned_bias_model`.
    """

    predictors: List[str]
    grid_points: List[np.ndarray]
    grid_values: np.ndarray
    factor_cols: List[str] = field(default_factory=list)
    factor_offsets: Dict[str, Dict[str, float]] = field(default_factory=dict)
    predictor_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    n_filled: int = 0
    n_observed: int = 0
    bin_table: Optional[pd.DataFrame] = None
    edges: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from scipy.interpolate import RegularGridInterpolator  # lazy

        self._interp = RegularGridInterpolator(
            tuple(np.asarray(g, dtype=float) for g in self.grid_points),
            np.asarray(self.grid_values, dtype=float),
            method="linear", bounds_error=False, fill_value=None,
        )

    def _clip(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float, copy=True)
        for j, p in enumerate(self.predictors):
            rng = self.predictor_ranges.get(p)
            if rng is not None and np.isfinite(rng[0]) and np.isfinite(rng[1]):
                X[:, j] = np.clip(X[:, j], rng[0], rng[1])
        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row μ (NaN where any continuous predictor is non-finite)."""
        missing = [p for p in self.predictors if p not in df.columns]
        if missing:
            raise KeyError(f"Predictors missing from frame: {missing}")
        X = df[self.predictors].to_numpy(dtype=float)
        ok = np.isfinite(X).all(axis=1)
        out = np.full(len(df), np.nan, dtype=float)
        if ok.any():
            out[ok] = self._interp(self._clip(X[ok]))
        for col in self.factor_cols:
            if col not in df.columns:
                continue
            table = self.factor_offsets.get(col, {})
            vals = df[col].astype(str).to_numpy()
            out = out + np.array([table.get(v, 0.0) for v in vals])
        return out


def fit_binned_bias_model(
    df: pd.DataFrame,
    predictors: Sequence[str] = ("slope",),
    factor_cols: Sequence[str] = (),
    *,
    n_bins: int = 10,
    min_count: int = 100,
    fac_spread_outliers: Optional[float] = 7.0,
    outlier_scope: str = "local",
    value_col: str = "dh",
    edges: Optional[Dict[str, np.ndarray]] = None,
) -> BinnedBiasModel:
    """Fit μ(x), the predictor-dependent robust centre of ``dh`` on stable terrain.

    **This is a diagnostic. Prefer not to apply it.**  Subtracting a fitted
    μ(x) changes the measured change, not just its uncertainty, and it is
    extrapolation wherever the feature of interest occupies terrain the stable
    area does not span; a μ(slope) fitted on gentle stable ground and applied
    to a steep landslide is exactly the case where it does most damage.  Where a
    terrain-dependent bias turns out to be an alignment residual (check with
    :func:`misregistration_diagnostic`), the right fix is upstream
    coregistration, not a post-hoc σ-side correction.  Where it is the error
    signature you set out to measure, correcting it away defeats the purpose.

    Use it in one of two ways, and say which one you used:

    **As a diagnostic (the recommended default).** Fit it, plot ``μ`` against each
    predictor, and compare its range to the σ field.  If ``max|μ − median dh|``
    is a meaningful fraction of σ, the difference raster carries terrain-
    correlated bias that neither the global median removal nor the σ model
    accounts for, and any σ fitted on top of it will be inflated.  Report that.

    **As a correction.** Subtract ``μ(x)`` from ``dh`` before fitting σ.  This is
    a terrain-bias deramp of the same family as ``xdem.coreg.TerrainBias`` /
    ``DirectionalBias`` and it genuinely improves the σ model, but it also
    changes the measured change, so it must be justified physically (a known
    horizontal misregistration or scan-geometry distortion) rather than applied
    because it improves a residual statistic.  It is only defensible when the
    stable terrain is representative of the feature of interest in that
    predictor: a μ(slope) fitted on flat stable ground and applied to a steep
    landslide is extrapolation.

    Parameters mirror :func:`fit_binned_sigma_model`.  Keep ``predictors`` to
    one or two; a bias surface fitted on many predictors will start absorbing
    real change.

    Returns
    -------
    BinnedBiasModel
    """
    predictors = [
        p for p in predictors
        if p in df.columns and np.isfinite(pd.to_numeric(df[p], errors="coerce")).any()
    ]
    if not predictors:
        raise ValueError("No usable continuous predictors after NaN screening.")
    factor_cols = [c for c in factor_cols if c in df.columns]

    work = df.reset_index(drop=True).copy()
    residual = work[value_col].to_numpy(dtype=float).copy()
    offsets: Dict[str, Dict[str, float]] = {}
    for col in factor_cols:
        base = robust_center(residual)
        table: Dict[str, float] = {}
        levels = work[col].astype(str).to_numpy()
        for lvl in pd.unique(levels):
            sel = levels == lvl
            if int(sel.sum()) < 30:
                continue
            table[str(lvl)] = float(robust_center(residual[sel]) - base)
        offsets[col] = table
        residual = residual - np.array([table.get(v, 0.0) for v in levels])
    work[value_col] = residual

    table_df, use_edges = nd_binning(
        work, predictors, value_col=value_col, statistic=robust_center,
        n_bins=n_bins, min_count=min_count,
        fac_spread_outliers=fac_spread_outliers, outlier_scope=outlier_scope,
        edges=edges,
    )

    grid_points = [0.5 * (use_edges[p][:-1] + use_edges[p][1:]) for p in predictors]
    grid = np.full(tuple(len(g) for g in grid_points), np.nan, dtype=float)
    idx = tuple(table_df[p + "_bin"].to_numpy(dtype=int) for p in predictors)
    stat = table_df["statistic"].to_numpy(dtype=float)
    good = np.isfinite(stat)
    if not good.any():
        raise ValueError(
            "No bin reached min_count. Lower min_count or reduce n_bins."
        )
    grid[tuple(i[good] for i in idx)] = stat[good]
    grid, n_filled = _fill_grid(grid, grid_points)

    return BinnedBiasModel(
        predictors=list(predictors),
        grid_points=grid_points,
        grid_values=grid,
        factor_cols=list(factor_cols),
        factor_offsets=offsets,
        predictor_ranges={p: (float(g.min()), float(g.max()))
                          for p, g in zip(predictors, grid_points)},
        n_filled=n_filled,
        n_observed=int(good.sum()),
        bin_table=table_df,
        edges=use_edges,
    )


def misregistration_diagnostic(
    df: pd.DataFrame,
    *,
    dh_col: str = "dh",
    slope_col: str = "slope",
    aspect_col: str = "aspect",
    min_slope_deg: float = 5.0,
    slope_poly: int = 2,
    max_abs_dh: Optional[float] = None,
) -> Dict[str, Any]:
    """Separate a horizontal-misregistration bias from a slope-only bias.

    A horizontal shift between two DEMs makes the elevation difference depend on
    terrain in a specific way (Nuth & Kääb, 2011)::

        dh = a · cos(b − aspect) · tan(slope)

    with ``a`` the shift magnitude and ``b`` its direction.  The distinguishing
    feature is the **aspect** dependence: a rigid shift raises one side of every
    ridge and lowers the other, whereas scan-geometry distortion, a vertical
    scale error or interpolation error under canopy bias both sides the same
    way.  Telling them apart matters because they call for different responses:
    the first is a coregistration residual, the second is an error property of
    the survey.

    Why this belongs in an uncertainty notebook: an unmodelled bias does not
    stay in the first moment.  Standardising ``dh / σ(x)`` when the true centre
    is not zero pushes the offset into the dispersion, so σ inflates to absorb
    it.  Knowing how much of your σ is absorbed bias (and which kind) is a
    precondition for interpreting the map, whether or not you correct anything.

    **This function only reports. It applies no correction.**

    The fit is::

        dh = a·cos(b − aspect)·tan(slope) + Σₖ cₖ·slopeᵏ

    i.e. the shift term is estimated *jointly with* a polynomial slope nuisance
    term rather than on its own.  That joint fit is necessary, not fastidious:
    in valley terrain slope and aspect are correlated (steep ground faces a
    preferred direction), and the plain Nuth & Kääb regression then attributes a
    purely slope-dependent bias to a shift that is not there.  Measured on
    synthetic terrain with that coupling and a planted slope-only bias, the
    unaugmented fit reported a spurious 0.62 m shift; with the slope term
    included it reports ~0.

    Parameters
    ----------
    df : pandas.DataFrame
        Stable-terrain frame with ``dh``, ``slope`` (degrees) and ``aspect``
        (degrees clockwise from north).
    min_slope_deg : float
        Slopes below this are dropped: the shift signal scales with
        ``tan(slope)`` and vanishes on flat ground, which only adds noise.
    slope_poly : int
        Degree of the slope nuisance polynomial.  ``0`` reduces to a plain
        intercept, i.e. the classical Nuth & Kääb form.
    max_abs_dh : float, optional
        Drop rows with ``|dh|`` above this (blunder guard).  Defaults to
        7 × NMAD.

    Returns
    -------
    dict
        ``shift_m``, ``direction_deg`` (azimuth of the shift, degrees
        counter-clockwise from east), ``vertical_m``, ``n``,
        ``aspect_bias_amplitude`` (the peak-to-peak bias the shift induces at the
        95th-percentile slope; evaluating it at the maximum slope would quote
        a number set by a handful of near-vertical pixels),
        ``slope_bias_span`` (peak-to-peak of the fitted slope-only term over
        the same range), ``frac_aspect`` (share of the explained bias variance
        carried by the aspect term), ``r2`` and ``aspect_slope_coupling``
        (|Spearman| between slope and cos(aspect); above ~0.3 the two terms are
        hard to separate and both numbers should be read as indicative).

    Reading it:

    - **``frac_aspect`` high, ``shift_m`` a meaningful fraction of the pixel
      size**: a coregistration residual.  It is not an error property of the
      survey, so a σ model fitted over it is partly measuring your alignment.
      Either coregister and re-run, or state that the reported σ includes it.
    - **``frac_aspect`` low with a large ``slope_bias_span``**: the bias varies
      with slope but not aspect, so it is not a rigid shift.  Scan-geometry
      distortion, a vertical scale or datum error, or slope-dependent
      processing error are the candidates.  This is often the thing the study
      set out to measure, in which case correcting it away defeats the purpose.

    References
    ----------
    Nuth, C., & Kääb, A. (2011). Co-registration and bias corrections of
        satellite elevation data sets for quantifying glacier thickness change.
        *The Cryosphere*, 5(1), 271–290. https://doi.org/10.5194/tc-5-271-2011
    """
    for c in (dh_col, slope_col, aspect_col):
        if c not in df.columns:
            raise KeyError(f"{c!r} required for the misregistration diagnostic")

    nan = {"shift_m": np.nan, "direction_deg": np.nan, "vertical_m": np.nan,
           "n": 0, "aspect_bias_amplitude": np.nan, "slope_bias_span": np.nan,
           "frac_aspect": np.nan, "r2": np.nan, "aspect_slope_coupling": np.nan}

    d = df[dh_col].to_numpy(dtype=float)
    sl_deg = df[slope_col].to_numpy(dtype=float)
    asp_deg = df[aspect_col].to_numpy(dtype=float)
    fin = np.isfinite(d) & np.isfinite(sl_deg) & np.isfinite(asp_deg)
    if fin.sum() < 100:
        return dict(nan, n=int(fin.sum()))
    d = d - robust_center(d[fin])

    if max_abs_dh is None:
        max_abs_dh = 7.0 * float(nmad(d[fin]))
    use = fin & (sl_deg >= min_slope_deg) & (np.abs(d) <= max_abs_dh)
    if use.sum() < 100:
        return dict(nan, n=int(use.sum()))

    sl = np.radians(sl_deg[use])
    asp = np.radians(asp_deg[use])
    tan_s = np.tan(sl)
    y = d[use]

    # Design: [cos(asp)*tan(s), sin(asp)*tan(s), 1, s, s^2, ...]
    cols = [np.cos(asp) * tan_s, np.sin(asp) * tan_s, np.ones(use.sum())]
    s_scaled = sl_deg[use] / max(float(np.max(sl_deg[use])), 1e-9)
    for k in range(1, int(slope_poly) + 1):
        cols.append(s_scaled ** k)
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)

    shift = float(np.hypot(coef[0], coef[1]))
    direction = float(np.degrees(np.arctan2(coef[1], coef[0])) % 360.0)

    aspect_part = A[:, :2] @ coef[:2]
    slope_part = A[:, 2:] @ coef[2:]
    fitted = aspect_part + slope_part
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum((y - fitted) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    va, vs = float(np.var(aspect_part)), float(np.var(slope_part))
    frac_aspect = float(va / (va + vs)) if (va + vs) > 0 else np.nan

    from scipy.stats import spearmanr  # lazy
    rho = spearmanr(sl_deg[use], np.cos(asp)).statistic
    coupling = float(abs(rho)) if np.isfinite(rho) else np.nan

    return {
        "shift_m": shift,
        "direction_deg": direction,
        "vertical_m": float(coef[2]),
        "n": int(use.sum()),
        "aspect_bias_amplitude": float(2.0 * shift * np.quantile(tan_s, 0.95)),
        "slope_bias_span": float(np.max(slope_part) - np.min(slope_part)),
        "frac_aspect": frac_aspect,
        "r2": r2,
        "aspect_slope_coupling": coupling,
    }


# ---------------------------------------------------------------------------
# Interoperability: wrap a foreign sigma function, reproduce a foreign binning
# ---------------------------------------------------------------------------

@dataclass
class CallableSigmaModel:
    """Adapter putting an arbitrary σ function behind the ``.predict(df)`` interface.

    ``fun`` takes a tuple of 1-D predictor arrays (in ``predictors`` order) and
    returns σ for each element.  That is exactly the signature of the error
    function returned by ``xdem.spatialstats.interp_nd_binning`` and
    ``infer_heteroscedasticity_from_stable``, so this adapter lets a *foreign*
    σ estimator be scored by :func:`cross_validate_sigma_models`, standardised
    by :func:`topochange.heteroscedastic.standardize` and rasterised by
    :func:`topochange.heteroscedastic.evaluate_sigma_raster` on exactly the same
    footing as this package's own models, which is the only way to compare two
    implementations honestly.

    It also covers the simpler case of a σ law you want to impose by hand
    (a vendor-specified accuracy curve, a published slope–σ relation).

    Nothing here imports the foreign package; only the callable is needed.

    Parameters
    ----------
    fun : callable
        ``fun((x1, x2, ...)) -> ndarray``.
    predictors : sequence of str
        Column names supplying the arrays, in the order ``fun`` expects.
    factor_cols, factor_offsets
        Optional categorical log-offsets applied multiplicatively on top, for
        parity with the other models here.  Usually empty.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> m = CallableSigmaModel(lambda v: 0.05 + 0.002 * np.asarray(v[0]), ["slope"])
    >>> float(m.predict(pd.DataFrame({"slope": [10.0]}))[0])
    0.07
    """

    fun: Any
    predictors: List[str]
    factor_cols: List[str] = field(default_factory=list)
    factor_offsets: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        missing = [p for p in self.predictors if p not in df.columns]
        if missing:
            raise KeyError(f"Predictors missing from frame: {missing}")
        cols = [df[p].to_numpy(dtype=float) for p in self.predictors]
        ok = np.ones(len(df), dtype=bool)
        for c in cols:
            ok &= np.isfinite(c)
        out = np.full(len(df), np.nan, dtype=float)
        if ok.any():
            out[ok] = np.asarray(self.fun(tuple(c[ok] for c in cols)), dtype=float).ravel()
        for col in self.factor_cols:
            if col not in df.columns:
                continue
            table = self.factor_offsets.get(col, {})
            vals = df[col].astype(str).to_numpy()
            out = out * np.exp(np.array([table.get(v, 0.0) for v in vals]))
        return out


def uniform_edges(
    df: pd.DataFrame, predictors: Sequence[str], n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """Equal-width bin edges spanning each predictor's range.

    Supply the result as :func:`fit_binned_sigma_model`'s ``edges`` to reproduce
    ``xdem.spatialstats.nd_binning``, which passes an integer ``bins`` to
    ``scipy.stats.binned_statistic_dd`` and therefore bins uniformly.

    This is a reproduction aid, not a recommendation.  Uniform edges are a poor
    fit for the predictors used here because they are all right-skewed: on the
    CA geometric-distortion scene a 10-bin uniform grid on the CHM puts 73% of
    the stable pixels in the first bin, while the top four bins hold 76 to 643
    pixels each, enough to pass a ``min_count`` of 100 in two of them and drive
    the interpolant with what is almost certainly residual real change rather
    than error.  Quantile edges (the default) keep the counts balanced.
    """
    out: Dict[str, np.ndarray] = {}
    for p in predictors:
        v = pd.to_numeric(df[p], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ValueError(f"No finite values for predictor {p!r}")
        e = np.linspace(float(v.min()), float(v.max()), int(n_bins) + 1)
        e[0] = np.nextafter(e[0], -np.inf)
        e[-1] = np.nextafter(e[-1], np.inf)
        out[p] = e
    return out


def trimmed_nmad_scale(z: np.ndarray, fac_spread_outliers: Optional[float] = 7.0) -> float:
    """The second-step standardisation factor, computed after trimming outliers.

    :func:`topochange.heteroscedastic.standardize` sets its ``sigma_scale`` to
    ``NMAD(z)`` over every row.  ``xdem.spatialstats.two_step_standardization``
    first masks ``|z| > fac_spread_outliers * NMAD(z)`` and only then recomputes
    the factor.  On clean data the two agree; on a stable mask that retains real
    change the trimmed factor is smaller, so xDEM's σ sits slightly lower.

    Use it to reconcile the two, or to get an xDEM-consistent level::

        scale_trimmed = trimmed_nmad_scale(std["z"].to_numpy())
        sigma_xdem_like = std["sigma_hat"] * scale_trimmed

    Returns ``nan`` if ``z`` has no finite entries.
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return float("nan")
    s = float(nmad(z))
    if fac_spread_outliers is None or not np.isfinite(s) or s <= 0:
        return s
    keep = np.abs(z) <= fac_spread_outliers * s
    if keep.sum() < 10:
        return s
    return float(nmad(z[keep]))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def write_sigma_geotiff(
    array: np.ndarray,
    path: str,
    transform,
    crs,
    *,
    nodata: float = -9999.0,
    dtype: str = "float32",
    description: str = "per-pixel 1-sigma elevation-difference uncertainty (m)",
    tags: Optional[Dict[str, str]] = None,
    compress: str = "lzw",
) -> str:
    """Write a per-pixel σ field to a tiled, compressed GeoTIFF.

    Non-finite pixels (no coverage, or a predictor that was undefined there) are
    written as ``nodata`` rather than NaN so the file reads correctly in GIS
    software that ignores NaN nodata declarations.

    Parameters
    ----------
    array : ndarray
        2-D σ field, e.g. ``run_heteroscedastic_pipeline(...)["sigma_raster"]``
        or :func:`topochange.heteroscedastic.evaluate_sigma_raster` output.
    path : str
        Output path.
    transform : affine.Affine
        Affine transform of the difference raster the σ field was built on.
    crs : rasterio CRS or str
        Coordinate reference system.
    nodata : float
        Nodata value stamped into non-finite cells.
    tags : dict, optional
        Extra GeoTIFF metadata tags (provenance: predictors used, model type,
        sigma_scale, source difference raster ...). Values are coerced to str.

    Returns
    -------
    str
        The path written.
    """
    import rasterio  # lazy

    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {arr.shape}")
    out = np.where(np.isfinite(arr), arr, nodata).astype(dtype)

    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": compress,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, description)
        meta = {"UNITS": "m", "STATISTIC": "1-sigma (NMAD-calibrated)"}
        if tags:
            meta.update({str(k): str(v) for k, v in tags.items()})
        dst.update_tags(**meta)
    return path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def mixture_nmad(sigmas: np.ndarray) -> float:
    """NMAD a zero-centred Gaussian *scale mixture* with these σ would exhibit.

    Needed to make a per-bin calibration ratio meaningful.  Within any bin the
    per-pixel σ still varies, and the NMAD of the pooled differences is **not**
    the median (or mean) of the σ values: a mixture of narrow and wide
    Gaussians is heavier-tailed than either, so ``NMAD(dh) / median(σ̂)``
    exceeds 1 even for a perfectly calibrated model.  Ignoring this makes every
    bin look under-predicted by a few percent.

    For a symmetric zero-centred mixture the median is 0 and
    ``median(|X|) = m`` solves ``(1/n) Σᵢ erf(m / (σᵢ √2)) = 1/2``; the returned
    value is ``1.4826 · m``, which reduces exactly to ``σ`` when all σ are equal.
    """
    from scipy.optimize import brentq  # lazy
    from scipy.special import erf  # lazy

    s = np.asarray(sigmas, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return float("nan")
    if np.allclose(s, s[0]):
        return float(s[0])
    if s.size > 20_000:  # the root solve only needs the σ distribution
        s = np.quantile(s, np.linspace(0.0, 1.0, 20_000))

    def f(m: float) -> float:
        return float(np.mean(erf(m / (s * np.sqrt(2.0))))) - 0.5

    lo, hi = 1e-12, 10.0 * float(s.max())
    if f(hi) < 0:  # pathological; fall back to the median σ
        return float(np.median(s))
    m = brentq(f, lo, hi, xtol=1e-12, rtol=1e-10)
    return float(1.4826 * m)


def calibration_by_bin(
    df: pd.DataFrame,
    predictor: str,
    *,
    dh_col: str = "dh",
    sigma_col: str = "sigma_hat",
    z_col: Optional[str] = "z",
    n_bins: int = 10,
    min_count: int = 50,
    center: Optional[float] = None,
) -> pd.DataFrame:
    """Observed-vs-predicted dispersion within bins of one predictor.

    This is the diagnostic that a global ``NMAD(z) = 1`` cannot provide.  The
    second step of :func:`topochange.heteroscedastic.standardize` rescales σ so
    that the *global* NMAD of z is exactly 1 by construction; the question that
    remains is whether it is also ≈ 1 within each slice of predictor space.
    Where the ratio departs from 1, the σ model is mis-shaped in that regime:
    ``> 1`` means σ is under-predicted (error there is larger than modelled),
    ``< 1`` means it is over-predicted.

    Parameters
    ----------
    df : pandas.DataFrame
        A standardised frame (output of
        :func:`topochange.heteroscedastic.standardize`).
    predictor : str
        Column to slice on.  Categorical (object/category) columns are grouped
        by level; numeric columns are quantile-binned.
    center : float, optional
        Centre subtracted from ``dh`` before measuring dispersion. Defaults to
        ``df.attrs['z_center']`` if present, else the robust centre of ``dh``.

    Returns
    -------
    pandas.DataFrame
        Columns ``bin`` (centre or level), ``count``, ``observed_nmad``
        (NMAD of centred ``dh``), ``predicted_sigma`` (median σ̂, for reading
        off the magnitude), ``predicted_nmad`` (:func:`mixture_nmad` of the
        bin's σ̂ values; the quantity ``observed_nmad`` should actually match),
        ``ratio = observed_nmad / predicted_nmad`` (target 1), ``nmad_z``
        (NMAD of ``z`` within the bin, target 1), ``coverage_95`` (fraction
        with ``|z| <= 1.96``, target 0.95 under a Gaussian), and the *bias*
        columns ``observed_median`` (median centred ``dh`` in the bin) and
        ``median_z`` (its standardised counterpart, target 0).

    Read the bias columns first.  A ``median_z`` that drifts systematically
    across bins means the difference raster carries a predictor-dependent
    *offset*, not just a predictor-dependent variance; see
    :class:`BinnedBiasModel`.  Left unmodelled it is absorbed into σ, so the
    dispersion columns cannot be interpreted until it is accounted for.

    ``ratio`` and ``nmad_z`` measure the same thing by two routes and should
    agree; ``nmad_z`` additionally carries the global two-step rescale, so a
    systematic offset between the two columns means that rescale is doing real
    work (i.e. the σ model alone was globally mis-scaled).
    """
    if predictor not in df.columns:
        raise KeyError(f"{predictor!r} not in frame")
    if center is None:
        center = float(df.attrs.get("z_center", robust_center(
            df[dh_col].to_numpy(dtype=float))))

    col = df[predictor]
    is_cat = isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object

    if is_cat:
        labels = col.astype(str).to_numpy()
        groups = [(lvl, labels == lvl) for lvl in pd.unique(labels)]
    else:
        v = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
        e = _quantile_edges(v, n_bins)
        idx = np.clip(np.digitize(v, e, right=False) - 1, -1, len(e) - 2)
        idx = np.where(np.isfinite(v), idx, -1)
        groups = [
            (0.5 * (e[k] + e[k + 1]), idx == k) for k in range(len(e) - 1)
        ]

    dh = df[dh_col].to_numpy(dtype=float) - center
    sig = df[sigma_col].to_numpy(dtype=float)
    z = df[z_col].to_numpy(dtype=float) if (z_col and z_col in df.columns) else dh / sig

    rows = []
    for label, sel in groups:
        sel = sel & np.isfinite(dh) & np.isfinite(sig) & (sig > 0)
        n = int(sel.sum())
        if n < min_count:
            continue
        obs = float(nmad(dh[sel]))
        pred = float(np.nanmedian(sig[sel]))
        pred_nmad = mixture_nmad(sig[sel])
        zz = z[sel]
        zz = zz[np.isfinite(zz)]
        rows.append({
            "bin": label,
            "count": n,
            "observed_nmad": obs,
            "predicted_sigma": pred,
            "predicted_nmad": pred_nmad,
            "ratio": obs / pred_nmad if pred_nmad > 0 else np.nan,
            "observed_median": float(np.median(dh[sel])),
            "median_z": float(np.median(zz)) if zz.size else np.nan,
            "nmad_z": float(nmad(zz)) if zz.size else np.nan,
            "coverage_95": float(np.mean(np.abs(zz) <= 1.959964)) if zz.size else np.nan,
        })
    return pd.DataFrame(rows)


def spatial_block_folds(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    block_size: float = 500.0,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Assign rows to spatially *blocked* cross-validation folds.

    Random row-wise k-fold is invalid here: adjacent pixels of a difference
    raster are strongly autocorrelated (that is the whole premise of the
    variogram step), so a random held-out pixel almost always has several of its
    own neighbours in the training set and the model scores far better than it
    would on genuinely new terrain.  Blocking at ``block_size`` (chosen at or
    above the correlation range) removes that leakage.

    Parameters
    ----------
    x, y : ndarray
        Pixel-centre map coordinates.
    n_folds : int
        Number of folds.
    block_size : float
        Block edge length in CRS units. Set it ≳ the long correlation range
        from the variogram step.
    seed : int, optional
        Seed for the block -> fold assignment.

    Returns
    -------
    ndarray of int
        Fold index in ``[0, n_folds)`` per row.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    bx = np.floor((x - np.nanmin(x)) / block_size).astype(np.int64)
    by = np.floor((y - np.nanmin(y)) / block_size).astype(np.int64)
    nbx = int(bx.max()) + 1
    block_id = by * nbx + bx
    uniq = np.unique(block_id)
    rng = np.random.default_rng(seed)
    assign = rng.permutation(len(uniq)) % int(n_folds)
    lookup = dict(zip(uniq.tolist(), assign.tolist()))
    return np.array([lookup[b] for b in block_id], dtype=int)


def _score_fold(dh: np.ndarray, sigma: np.ndarray, df_test: pd.DataFrame,
                predictors: Sequence[str], center: float,
                n_bins: int, min_count: int) -> Dict[str, float]:
    """Held-out scores for one (dh, σ̂) pair."""
    ok = np.isfinite(dh) & np.isfinite(sigma) & (sigma > 0)
    if ok.sum() < 50:
        return {"n": int(ok.sum()), "nmad_z": np.nan, "nll": np.nan,
                "nll_median": np.nan, "calibration_rmse": np.nan,
                "coverage_95": np.nan, "k_cover_95": np.nan,
                "frac_predicted": float(ok.mean()) if ok.size else 0.0}
    r = dh[ok] - center
    s = sigma[ok]
    z = r / s

    # Gaussian negative log-likelihood: a strictly proper scoring rule for a
    # scale parameter, so it rewards getting sigma right pixel-by-pixel rather
    # than only on average. Constant terms dropped.
    pointwise = np.log(s) + 0.5 * (z ** 2)
    nll = float(np.mean(pointwise))
    # Robust companion. The mean NLL is unbounded above, so a handful of pixels
    # where an estimator predicted a near-zero sigma (the classic GAM
    # extrapolation failure) can dominate it entirely and produce scores in the
    # hundreds. The median is not a proper scoring rule and must not be the
    # selection criterion, but if it ranks the estimators the same way then the
    # ranking is not an artefact of those few pixels.
    nll_median = float(np.median(pointwise))

    # Calibration RMSE: root-mean-square of log(observed NMAD / predicted sigma)
    # over held-out predictor bins. This is the direct measure of whether the
    # *shape* of the heteroscedasticity transferred, and unlike nmad_z it cannot
    # be satisfied by a global rescale.
    tmp = df_test.loc[ok].copy()
    tmp["_r"] = r
    tmp["_s"] = s
    logs = []
    for p in predictors:
        if p not in tmp.columns:
            continue
        v = pd.to_numeric(tmp[p], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(v).any():
            continue
        try:
            e = _quantile_edges(v, n_bins)
        except ValueError:
            continue
        idx = np.clip(np.digitize(v, e, right=False) - 1, -1, len(e) - 2)
        for k in range(len(e) - 1):
            sel = idx == k
            if sel.sum() < min_count:
                continue
            obs = nmad(tmp["_r"].to_numpy()[sel])
            pred = mixture_nmad(tmp["_s"].to_numpy()[sel])
            if obs > 0 and np.isfinite(pred) and pred > 0:
                logs.append(np.log(obs / pred))
    cal = float(np.sqrt(np.mean(np.square(logs)))) if logs else np.nan

    return {
        "n": int(ok.sum()),
        "nmad_z": float(nmad(z)),
        "nll": nll,
        "nll_median": nll_median,
        "calibration_rmse": cal,
        "coverage_95": float(np.mean(np.abs(z) <= 1.959964)),
        # The empirical multiplier that would give 95% coverage. Real stable
        # terrain is heavy-tailed (residual change, blunders), so 1.96 usually
        # under-covers; this is the honest factor for a per-pixel interval.
        "k_cover_95": float(np.quantile(np.abs(z), 0.95)),
        "frac_predicted": float(ok.mean()),
    }


def cross_validate_sigma_models(
    df: pd.DataFrame,
    fitters: Dict[str, Callable[[pd.DataFrame], Any]],
    *,
    predictors: Sequence[str] = ("slope", "roughness"),
    n_folds: int = 5,
    block_size: float = 500.0,
    seed: Optional[int] = 0,
    dh_col: str = "dh",
    x_col: str = "x",
    y_col: str = "y",
    n_bins: int = 8,
    min_count: int = 50,
    rescale: bool = True,
) -> pd.DataFrame:
    """Spatially blocked cross-validation comparing σ estimators.

    Each fitter is trained on the training folds only and scored on the held-out
    fold, so a model that memorises the training terrain is penalised.  When
    ``rescale`` (default), the Hugonnet two-step factor ``NMAD(z_train)`` is
    computed on the *training* rows and applied to the test predictions; this
    keeps the comparison about heteroscedastic *shape* rather than global scale,
    and mirrors what :func:`topochange.heteroscedastic.standardize` does in
    production.

    Parameters
    ----------
    df : pandas.DataFrame
        Stable-terrain frame with ``dh``, ``x``, ``y`` and the predictors.
    fitters : dict
        ``{name: callable(train_df) -> model}``; the model needs only
        ``.predict(df) -> ndarray``.  A fitter that raises on a fold records
        NaN scores for that fold instead of aborting the comparison.
    predictors : sequence of str
        Predictors used for the per-bin calibration score.
    n_folds, block_size, seed
        Passed to :func:`spatial_block_folds`.

    Returns
    -------
    pandas.DataFrame
        One row per (model, fold) with ``n``, ``nmad_z`` (target 1),
        ``nll`` (Gaussian negative log-likelihood, **lower is better**; the
        primary criterion), ``nll_median`` (its robust companion; see
        ``_score_fold``), ``calibration_rmse`` (target 0), ``coverage_95``
        (target 0.95 under a Gaussian), ``k_cover_95`` (the empirical
        multiplier that actually delivers 95% coverage; use it instead of
        1.96 for a per-pixel interval), ``frac_predicted`` (share of test rows
        the model could score at all; a model needing a predictor that is
        often NaN is penalised here) and ``error`` (the exception message, if
        the fit failed).
    """
    for c in (dh_col, x_col, y_col):
        if c not in df.columns:
            raise KeyError(f"{c!r} required for cross-validation")

    work = df.reset_index(drop=True)
    folds = spatial_block_folds(
        work[x_col].to_numpy(), work[y_col].to_numpy(),
        n_folds=n_folds, block_size=block_size, seed=seed,
    )

    rows = []
    for name, fitter in fitters.items():
        for k in range(n_folds):
            train = work.loc[folds != k]
            test = work.loc[folds == k]
            if len(train) < 500 or len(test) < 100:
                continue
            rec: Dict[str, Any] = {"model": name, "fold": k}
            try:
                model = fitter(train)
                s_train = np.asarray(model.predict(train), dtype=float)
                s_test = np.asarray(model.predict(test), dtype=float)
            except Exception as exc:  # a fitter can legitimately fail on a fold
                rec.update({"n": 0, "nmad_z": np.nan, "nll": np.nan,
                            "nll_median": np.nan, "calibration_rmse": np.nan,
                            "coverage_95": np.nan, "k_cover_95": np.nan,
                            "frac_predicted": 0.0,
                            "error": f"{type(exc).__name__}: {exc}"})
                rows.append(rec)
                continue

            dh_train = train[dh_col].to_numpy(dtype=float)
            centre = robust_center(dh_train[np.isfinite(dh_train)])

            # Training-side floor + two-step rescale, exactly as standardize()
            # does, but estimated on the training rows only.
            pos = s_train[np.isfinite(s_train) & (s_train > 0)]
            floor = max(1e-6, 0.05 * float(np.median(pos))) if pos.size else 1e-6
            s_train = np.where(np.isfinite(s_train), np.maximum(s_train, floor), np.nan)
            s_test = np.where(np.isfinite(s_test), np.maximum(s_test, floor), np.nan)
            if rescale:
                zt = (dh_train - centre) / s_train
                zt = zt[np.isfinite(zt)]
                if zt.size:
                    f = float(nmad(zt))
                    if np.isfinite(f) and f > 0:
                        s_test = s_test * f

            rec.update(_score_fold(
                test[dh_col].to_numpy(dtype=float), s_test, test,
                predictors, centre, n_bins, min_count,
            ))
            rec["error"] = ""
            rows.append(rec)

    return pd.DataFrame(rows)


def patch_validation(
    dh: np.ndarray,
    transform,
    *,
    stable_mask: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    areas: Sequence[float] = (100.0, 1_000.0, 10_000.0, 100_000.0),
    n_patches: int = 500,
    min_valid_frac: float = 0.9,
    seed: Optional[int] = 0,
    center: Optional[float] = None,
) -> pd.DataFrame:
    """Empirical dispersion of area means on stable terrain, by patch area.

    The Monte-Carlo patch method of Hugonnet et al. (2022, §V-D2), as
    implemented by ``xdem.spatialstats.patches_method``: draw square patches of
    a given area on stable terrain, take the mean ``dh`` in each, and report the
    NMAD *across* patch means.  Because the true change over stable terrain is
    zero, that NMAD is a direct, model-free estimate of the standard error of an
    area mean at that scale.

    Plotted against area it is the single strongest available check on the whole
    chain: an areal-σ curve derived from the σ field plus the fitted correlogram
    should track it.  If the modelled curve sits below the empirical one, the
    correlation range is being under-estimated (the classic failure); if the
    ``sqrt(mean σ²)/√N`` curve is used instead, the gap between the two is
    exactly the variance that spatial correlation contributes.

    Parameters
    ----------
    dh : ndarray
        2-D difference raster (non-finite = no data).
    transform : affine.Affine
        Its affine transform (used for the pixel area).
    stable_mask : ndarray, optional
        Boolean/0-1 stable-terrain mask; patches are drawn only where True.
    sigma : ndarray, optional
        Per-pixel σ field.  When given, the returned frame also carries the
        standard error you would predict if the error were spatially
        *independent*: per patch ``sqrt(mean(σ²)/n)``, aggregated across
        patches both as an NMAD-consistent value (:func:`mixture_nmad`, paired
        with ``empirical_sigma_mean``) and as an RMS (paired with
        ``empirical_sigma_mean_std``).  The ratio of observed to predicted is
        the correlation inflation factor at that scale: ~1 means the error
        really is independent at that scale; large values mean spatial
        correlation dominates, which is the usual finding and the reason an
        areal σ cannot be obtained by dividing a σ map by √N.
    areas : sequence of float
        Patch areas in squared CRS units.
    n_patches : int
        Patches attempted per area.
    min_valid_frac : float
        A patch is used only if this fraction of its cells are valid (and
        stable, when a mask is given).
    center : float, optional
        Value subtracted from ``dh`` first (defaults to the robust centre of the
        stable pixels, the same datum the rest of the workflow removes).

    Returns
    -------
    pandas.DataFrame
        ``area``, ``patch_size_px``, ``n_patches`` (accepted), ``n_pixels``
        (mean valid cells per patch), ``empirical_sigma_mean`` (NMAD across
        patch means), ``empirical_sigma_mean_std`` (its standard-deviation
        counterpart), ``mean_of_means`` and, when ``sigma`` is given,
        ``uncorrelated_prediction_rms`` and
        ``uncorrelated_prediction`` and ``correlation_inflation``.
    """
    dh = np.asarray(dh, dtype=float)
    if dh.ndim != 2:
        raise ValueError("dh must be 2-D")
    res_x = abs(float(transform.a))
    res_y = abs(float(transform.e))
    pix_area = res_x * res_y

    valid = np.isfinite(dh)
    if stable_mask is not None:
        sm = np.asarray(stable_mask)
        if sm.shape != dh.shape:
            raise ValueError("stable_mask shape must match dh")
        valid &= np.isfinite(sm.astype(float)) & (sm.astype(float) > 0)
    if not valid.any():
        raise ValueError("No valid stable pixels.")

    if center is None:
        center = float(robust_center(dh[valid]))
    work = np.where(valid, dh - center, np.nan)
    sig = np.asarray(sigma, dtype=float) if sigma is not None else None
    if sig is not None and sig.shape != dh.shape:
        raise ValueError("sigma shape must match dh")

    rng = np.random.default_rng(seed)
    ny, nx = dh.shape
    rows = []
    for area in areas:
        side = max(int(round(np.sqrt(float(area) / pix_area))), 1)
        if side > ny or side > nx:
            rows.append({"area": float(area), "patch_size_px": side, "n_patches": 0,
                         "n_pixels": np.nan, "empirical_sigma_mean": np.nan,
                         "empirical_sigma_mean_std": np.nan, "mean_of_means": np.nan})
            continue
        means, counts, unc = [], [], []
        attempts = 0
        max_attempts = int(n_patches) * 20
        while len(means) < int(n_patches) and attempts < max_attempts:
            attempts += 1
            r0 = int(rng.integers(0, ny - side + 1))
            c0 = int(rng.integers(0, nx - side + 1))
            block = work[r0:r0 + side, c0:c0 + side]
            good = np.isfinite(block)
            n_good = int(good.sum())
            if n_good < min_valid_frac * side * side:
                continue
            means.append(float(np.mean(block[good])))
            counts.append(n_good)
            if sig is not None:
                sb = sig[r0:r0 + side, c0:c0 + side][good]
                sb = sb[np.isfinite(sb)]
                if sb.size:
                    unc.append(float(np.sqrt(np.mean(sb ** 2)) / np.sqrt(sb.size)))
        m = np.asarray(means, dtype=float)
        rec = {
            "area": float(area),
            "patch_size_px": side,
            "n_patches": int(m.size),
            "n_pixels": float(np.mean(counts)) if counts else np.nan,
            "empirical_sigma_mean": float(nmad(m)) if m.size > 2 else np.nan,
            "empirical_sigma_mean_std": float(np.std(m, ddof=1)) if m.size > 2 else np.nan,
            "mean_of_means": float(np.mean(m)) if m.size else np.nan,
        }
        if sig is not None:
            u = np.asarray(unc, dtype=float)
            # The patches do not all have the same predicted SD (σ varies across
            # the scene), so the *distribution of patch means* is itself a scale
            # mixture. Comparing its NMAD against the mean predicted SD would
            # therefore read low even for a perfect model; mixture_nmad converts
            # the per-patch SDs into the NMAD they would actually produce.
            rec["uncorrelated_prediction"] = mixture_nmad(u) if u.size else np.nan
            rec["uncorrelated_prediction_rms"] = (
                float(np.sqrt(np.mean(u ** 2))) if u.size else np.nan
            )
            pred = rec["uncorrelated_prediction"]
            rec["correlation_inflation"] = (
                rec["empirical_sigma_mean"] / pred
                if u.size and np.isfinite(pred) and pred > 0 else np.nan
            )
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_1d_binning(
    table: pd.DataFrame,
    predictor: str,
    *,
    ax=None,
    label: Optional[str] = None,
    show_counts: bool = True,
):
    """Plot binned dispersion against one predictor (xDEM's ``plot_1d_binning``).

    ``table`` is a :func:`nd_binning` output.  If it has more than one binning
    axis, the other axes are collapsed by a count-weighted mean, so the curve is
    the marginal effect of ``predictor``.
    """
    import matplotlib.pyplot as plt  # lazy

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.4))
    t = table.dropna(subset=["statistic"]).copy()
    if t.empty:
        raise ValueError("No occupied bins to plot.")
    g = t.groupby(predictor, observed=True).apply(
        lambda s: pd.Series({
            "statistic": float(np.average(s["statistic"], weights=s["count"])),
            "count": float(s["count"].sum()),
        }),
        include_groups=False,
    ).reset_index()
    ax.plot(g[predictor], g["statistic"], "o-", color="#2f6f9f", label=label)
    ax.set_xlabel(predictor)
    ax.set_ylabel("NMAD of dh (m)")
    ax.grid(alpha=0.3)
    if show_counts:
        ax2 = ax.twinx()
        ax2.bar(g[predictor], g["count"], alpha=0.15, color="grey",
                width=np.diff(np.r_[g[predictor].values,
                                    g[predictor].values[-1] * 1.01]).clip(min=1e-9))
        ax2.set_ylabel("count", color="grey")
        ax2.set_yscale("log")
    if label:
        ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_2d_binning(
    table: pd.DataFrame,
    predictor_x: str,
    predictor_y: str,
    *,
    ax=None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Heat-map of binned dispersion over two predictors (xDEM's ``plot_2d_binning``)."""
    import matplotlib.pyplot as plt  # lazy

    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4.0))
    t = table.dropna(subset=["statistic"])
    piv = t.pivot_table(index=predictor_y, columns=predictor_x,
                        values="statistic", aggfunc="mean")
    im = ax.pcolormesh(piv.columns.values, piv.index.values, piv.values,
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_xlabel(predictor_x)
    ax.set_ylabel(predictor_y)
    plt.colorbar(im, ax=ax, label="NMAD of dh (m)")
    return ax


def plot_sigma_map(
    sigma: np.ndarray,
    *,
    transform=None,
    ax=None,
    cmap: str = "magma",
    percentile_clip: Tuple[float, float] = (2.0, 98.0),
    title: str = "per-pixel σ (m)",
):
    """Display a σ field with a robust colour stretch.

    A σ map is long-tailed almost by construction (a handful of pixels in the
    worst predictor corner dominate), so a raw min/max stretch renders the whole
    scene flat.  Default clip is the 2nd–98th percentile of the finite values.
    """
    import matplotlib.pyplot as plt  # lazy

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.0))
    a = np.asarray(sigma, dtype=float)
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        raise ValueError("σ field is entirely non-finite.")
    vmin, vmax = np.percentile(fin, percentile_clip)
    extent = None
    if transform is not None:
        h, w = a.shape
        x0, y0 = transform * (0, 0)
        x1, y1 = transform * (w, h)
        extent = (x0, x1, y1, y0)
    im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
                   interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8, label="σ (m)")
    return ax
