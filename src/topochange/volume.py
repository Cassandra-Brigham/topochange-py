"""Volumetric change and its uncertainty from a difference raster.

Converts the regionalized elevation-change uncertainty produced by
:class:`~topochange.uncertainty.RegionalUncertaintyEstimator` into a
volumetric uncertainty for a polygonal feature of interest.

Method
------
The net volume change over a polygon A is the integral of the vertical
difference dh over the polygon, estimated from the valid raster cells:

    V = Σᵢ dhᵢ · a          (a = cell area)

Because V = A · ⟨dh⟩, the standard error of V follows *linearly* from the
standard error of the spatial mean:

    σ_V = A · σ_ΔH

where σ_ΔH is the regionalized uncertainty of the mean elevation change over
the polygon, the quantity RegionalUncertaintyEstimator computes via Krige's
relation (Rolstad et al., 2009; Hugonnet et al., 2022).  Spatial correlation
is what makes this number honest: assuming independent pixels
(σ_V = a·σ₀·√N) underestimates σ_V dramatically for any realistic error
structure, while assuming full correlation (σ_V = A·σ₀) is the pessimistic
bound.  Both bounds are reported for context.

Thresholding (masking |dh| below a level of detection) is supported for
*gross* cut/fill volumes, where it removes spurious contributions of noise to
the totals, but is discouraged for *net* volumes, which it biases
(Anderson, 2019).  A warning is emitted if a threshold is applied to the net
volume.

References
----------
- Rolstad, C., Haug, T., & Denby, B. (2009). Spatially integrated geodetic
  glacier mass balance and its uncertainty based on geostatistical analysis.
  J. Glaciol., 55(192), 666-680. doi:10.3189/002214309789470950
- Hugonnet, R., et al. (2022). Uncertainty analysis of digital elevation
  models by spatial inference from stable terrain. IEEE JSTARS, 15,
  6456-6472. doi:10.1109/JSTARS.2022.3188922
- Anderson, S. W. (2019). Uncertainty in quantitative analyses of topographic
  change: error propagation and the role of thresholding. ESPL, 44(5),
  1015-1033. doi:10.1002/esp.4551
- Wheaton, J. M., Brasington, J., Darby, S. E., & Sear, D. A. (2010).
  Accounting for uncertainty in DEMs from repeat topographic surveys:
  improved sediment budgets. ESPL, 35(2), 136-156. doi:10.1002/esp.1886
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from rasterio.features import geometry_mask

from .uncertainty import RegionalUncertaintyEstimator


__all__ = ["polygon_volume", "VolumeResult", "VolumeEstimator"]


# ──────────────────────────────────────────────────────────────────────────
# Core volume integration (standalone, dependency-light, unit-testable)
# ──────────────────────────────────────────────────────────────────────────

def polygon_volume(
    arr: np.ndarray,
    transform,
    polygon: Union[Polygon, MultiPolygon],
    cell_area: float,
    lod: Optional[float] = None,
) -> Dict[str, float]:
    """Integrate a difference array over a polygon.

    Parameters
    ----------
    arr : ndarray (2D)
        Difference raster values (NaN = nodata).  Units: elevation units.
    transform : affine.Affine
        Raster geotransform (as from ``rasterio``/``rioxarray``).
    polygon : Polygon or MultiPolygon
        Feature of interest, in the raster CRS (projected, meters).
    cell_area : float
        Area of one cell (resolution², m²).
    lod : float, optional
        Level of detection.  Cells with |dh| < lod are excluded from the
        *gross* cut/fill sums (``v_cut_lod`` / ``v_fill_lod``).  The
        unthresholded net and gross volumes are always returned.

    Returns
    -------
    dict with keys:
        ``v_net``       net volume change (signed, elevation-unit · m²)
        ``v_cut``       gross lowering volume (positive magnitude)
        ``v_fill``      gross raising volume (positive)
        ``v_cut_lod``, ``v_fill_lod``  gross volumes after LoD masking
                        (equal to unthresholded values when ``lod`` is None)
        ``mean_dh``     mean difference over valid polygon cells
        ``n_valid``     number of valid cells inside the polygon
        ``n_inside``    number of cells inside the polygon (valid or not)
        ``area_valid``  n_valid · cell_area
        ``frac_valid``  n_valid / n_inside (data completeness inside polygon)
    """
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    arr = np.asarray(arr, dtype=float)

    inside = geometry_mask(
        [polygon], out_shape=arr.shape, transform=transform, invert=True
    )
    n_inside = int(inside.sum())
    if n_inside == 0:
        raise ValueError(
            "Polygon does not overlap the raster grid (0 cells inside). "
            "Check that the polygon is in the raster CRS."
        )

    valid = inside & np.isfinite(arr)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("No valid (finite) raster cells inside the polygon.")

    dh = arr[valid]

    v_net = float(dh.sum() * cell_area)
    v_cut = float(-dh[dh < 0].sum() * cell_area)
    v_fill = float(dh[dh > 0].sum() * cell_area)

    if lod is not None and lod > 0:
        keep = np.abs(dh) >= lod
        dh_t = dh[keep]
        v_cut_lod = float(-dh_t[dh_t < 0].sum() * cell_area)
        v_fill_lod = float(dh_t[dh_t > 0].sum() * cell_area)
    else:
        v_cut_lod, v_fill_lod = v_cut, v_fill

    return {
        "v_net": v_net,
        "v_cut": v_cut,
        "v_fill": v_fill,
        "v_cut_lod": v_cut_lod,
        "v_fill_lod": v_fill_lod,
        "mean_dh": float(dh.mean()),
        "n_valid": n_valid,
        "n_inside": n_inside,
        "area_valid": n_valid * cell_area,
        "frac_valid": n_valid / n_inside,
    }


# ──────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class VolumeResult:
    """Volumetric change with uncertainty over one polygon.

    All volumes are in (elevation unit) · m², i.e. m³ when the difference
    raster is in meters and the CRS is projected in meters.
    """

    # volumes
    v_net: float = np.nan
    v_cut: float = np.nan
    v_fill: float = np.nan
    v_cut_lod: float = np.nan
    v_fill_lod: float = np.nan

    # geometry / data support
    area_polygon: float = np.nan
    area_valid: float = np.nan
    area_basis: str = "valid"          # 'valid' or 'polygon'
    area_used: float = np.nan
    frac_valid: float = np.nan
    n_valid: int = 0
    cell_area: float = np.nan
    mean_dh: float = np.nan

    # uncertainty (volumes)
    sigma_v: float = np.nan            # total: quadrature of components below
    sigma_v_corr: float = np.nan       # A · sigma_A (correlated, Krige)
    sigma_v_uncorr: float = np.nan     # A · sigma_uncorrelated(mean)
    sigma_v_bias: Optional[float] = None   # A · sigma_a_stable (bias SE)
    include_bias_se: bool = False

    # interval bounds (from the sigma_A CI method, scaled by A)
    sigma_v_min: Optional[float] = None
    sigma_v_max: Optional[float] = None
    sigma_v_p025: Optional[float] = None
    sigma_v_p975: Optional[float] = None

    # pedagogical bounds
    sigma_v_naive: float = np.nan      # independent-pixel: a·σ₀·√N
    sigma_v_fullcorr: float = np.nan   # fully correlated: A·σ₀

    # provenance
    lod: Optional[float] = None
    ci_method: Optional[str] = None
    unit: str = "m"

    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extras"}
        d.update(self.extras)
        return d

    def summary(self) -> str:
        u = self.unit
        vu = f"{u}³" if u == "m" else f"{u}·m²"

        def _fmt(x, nd=1):
            return "—" if x is None or not np.isfinite(x) else f"{x:,.{nd}f}"

        lines = [
            "=" * 70,
            "VOLUMETRIC CHANGE SUMMARY",
            "=" * 70,
            f"Area basis: {self.area_basis}  "
            f"(polygon {_fmt(self.area_polygon)} m², "
            f"valid {_fmt(self.area_valid)} m², "
            f"completeness {self.frac_valid:.1%})",
            "",
            f"Net volume change : {_fmt(self.v_net)} {vu}  "
            f"± {_fmt(self.sigma_v)} {vu} (1σ)",
        ]
        if self.sigma_v_min is not None and self.sigma_v_max is not None:
            lines.append(
                f"  σ_V 68% interval : [{_fmt(self.sigma_v_min)}, "
                f"{_fmt(self.sigma_v_max)}] {vu}"
            )
        if self.sigma_v_p025 is not None and self.sigma_v_p975 is not None:
            lines.append(
                f"  σ_V 95% interval : [{_fmt(self.sigma_v_p025)}, "
                f"{_fmt(self.sigma_v_p975)}] {vu}"
            )
        lines += [
            "",
            f"Gross cut  (lowering): {_fmt(self.v_cut)} {vu}"
            + (f"   [LoD-masked: {_fmt(self.v_cut_lod)}]"
               if self.lod else ""),
            f"Gross fill (raising) : {_fmt(self.v_fill)} {vu}"
            + (f"   [LoD-masked: {_fmt(self.v_fill_lod)}]"
               if self.lod else ""),
            "",
            "Uncertainty budget (1σ, volume units):",
            f"  correlated (A·σ_A)     : {_fmt(self.sigma_v_corr)}",
            f"  uncorrelated           : {_fmt(self.sigma_v_uncorr)}",
        ]
        if self.sigma_v_bias is not None:
            folded = "folded into total" if self.include_bias_se else \
                     "reported alongside, NOT in total"
            lines.append(
                f"  bias SE (A·σ_A,stable) : {_fmt(self.sigma_v_bias)}  "
                f"[{folded}]"
            )
        lines += [
            "",
            "Context bounds:",
            f"  naive independent-pixel σ_V : {_fmt(self.sigma_v_naive)}  "
            f"(underestimate for correlated error)",
            f"  fully-correlated σ_V        : {_fmt(self.sigma_v_fullcorr)}  "
            f"(pessimistic bound)",
        ]
        if self.lod:
            lines.append(f"LoD threshold: |dh| ≥ {self.lod:g} {u} "
                         f"(applied to gross volumes only)")
        if self.ci_method:
            lines.append(f"σ_A interval method: {self.ci_method}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Estimator: bridges RegionalUncertaintyEstimator to volume
# ──────────────────────────────────────────────────────────────────────────

class VolumeEstimator:
    """Compute volumetric change ± uncertainty for a feature of interest.

    Wraps a :class:`RegionalUncertaintyEstimator` (which holds the difference
    raster, the polygon, the fitted variogram, and, after
    ``calc_total_uncertainty()``, the regionalized σ_A with interval bounds)
    and converts everything to volume via σ_V = A · σ_ΔH.

    Examples
    --------
    >>> est = RegionalUncertaintyEstimator(rdh, gv, area_of_interest=poly,
    ...                                    unstable_geoms=all_features)
    >>> est.calc_total_uncertainty(n_pairs=25_000, seed=42)
    >>> vol = VolumeEstimator(est).compute()
    >>> print(vol.summary())

    Notes
    -----
    *Area basis.*  With data gaps inside the polygon there are two
    conventions: ``area_basis='valid'`` (default) integrates over the valid
    cells only: V and σ_V describe the *observed* part of the feature;
    ``area_basis='polygon'`` extrapolates the valid-cell mean to the full
    polygon area: V and σ_V describe the whole feature under the assumption
    that the gaps behave like the observed cells.  The two coincide at 100%
    completeness.  A warning is emitted when completeness < 90% because gap
    extrapolation then carries additional (unmodeled) uncertainty.

    *Bias.*  The correlation-aware standard error of the stable-area median
    bias (``sigma_a_stable``) scales to volume as A·σ_A,stable.  Following
    the convention of ``calc_total_uncertainty``, it is reported alongside
    the total by default and folded in (quadrature) only when
    ``include_bias_se=True``.
    """

    def __init__(
        self,
        regional_estimator: RegionalUncertaintyEstimator,
        area_basis: str = "valid",
    ):
        if area_basis not in ("valid", "polygon"):
            raise ValueError("area_basis must be 'valid' or 'polygon'.")
        self.est = regional_estimator
        self.area_basis = area_basis

    # -- helpers ----------------------------------------------------------

    def _ensure_uncertainty(self, **kwargs) -> None:
        """Run calc_total_uncertainty() if it has not been run yet."""
        if self.est.total_uncertainty_polygon is None:
            self.est.calc_total_uncertainty(**kwargs)

    def _raster_pieces(self):
        rdh = self.est.raster_data_handler
        da = rdh.rioxarray_obj
        if da is None:
            raise RuntimeError(
                "RasterDataHandler has no loaded raster. Call load_raster() "
                "before computing volumes."
            )
        arr = da.values
        transform = da.rio.transform()
        res = float(rdh.resolution)
        return arr, transform, res

    # -- main entry point -------------------------------------------------

    def compute(
        self,
        lod: Optional[Union[float, str]] = None,
        include_bias_se: bool = False,
        threshold_net: bool = False,
        **uncertainty_kwargs,
    ) -> VolumeResult:
        """Compute volumes and volumetric uncertainty.

        Parameters
        ----------
        lod : float or 'Nsigma', optional
            Level of detection for the gross cut/fill volumes.  A float is an
            absolute |dh| threshold in elevation units; a string like
            ``'2sigma'`` uses N × σ₀ (the stable-area RMS of the difference
            raster).  Applied to GROSS volumes only unless ``threshold_net``.
        include_bias_se : bool
            Fold the bias SE term (A·σ_A,stable) into σ_V in quadrature.
            Default False, matching ``calc_total_uncertainty``.
        threshold_net : bool
            Also apply the LoD to the net volume.  Discouraged: thresholding
            net change produces biased estimates (Anderson, 2019).  A warning
            is emitted when True.
        **uncertainty_kwargs
            Forwarded to ``calc_total_uncertainty()`` if it has not yet been
            run on the regional estimator (e.g. ``n_pairs``, ``seed``,
            ``ci_method``).

        Returns
        -------
        VolumeResult
        """
        self._ensure_uncertainty(**uncertainty_kwargs)
        est = self.est

        arr, transform, res = self._raster_pieces()
        cell_area = res ** 2

        # resolve LoD
        lod_value: Optional[float] = None
        if lod is not None:
            if isinstance(lod, str):
                s = lod.lower().replace(" ", "")
                if not s.endswith("sigma"):
                    raise ValueError(
                        "String lod must look like '1sigma', '2sigma', ..."
                    )
                if est.sigma0_uncorrelated is None:
                    raise RuntimeError(
                        "σ₀ unavailable for a '*sigma' LoD — run "
                        "calc_total_uncertainty() first."
                    )
                k = float(s[:-5]) if s[:-5] else 1.0
                lod_value = k * float(est.sigma0_uncorrelated)
            else:
                lod_value = float(lod)

        vols = polygon_volume(arr, transform, est.polygon, cell_area,
                              lod=lod_value)

        v_net = vols["v_net"]
        if threshold_net and lod_value:
            warnings.warn(
                "Applying a level of detection to the NET volume biases the "
                "estimate toward zero net change and is discouraged "
                "(Anderson, 2019, doi:10.1002/esp.4551). Reporting the "
                "thresholded net volume as requested; the unthresholded net "
                "volume is kept in .extras['v_net_unthresholded'].",
                UserWarning, stacklevel=2,
            )
            v_net = vols["v_fill_lod"] - vols["v_cut_lod"]

        # area to scale σ_ΔH by
        area_polygon = float(est.area)
        area_valid = float(vols["area_valid"])
        area_used = area_valid if self.area_basis == "valid" else area_polygon

        if vols["frac_valid"] < 0.9:
            warnings.warn(
                f"Only {vols['frac_valid']:.1%} of the polygon has valid "
                f"data. Volumes on the '{self.area_basis}' basis "
                + ("describe the observed cells only."
                   if self.area_basis == "valid" else
                   "extrapolate the observed mean over the gaps, which adds "
                   "unmodeled uncertainty."),
                UserWarning, stacklevel=2,
            )

        if self.area_basis == "polygon" and not (threshold_net and lod_value):
            # extrapolate the valid-cell mean over the full polygon.
            # When threshold_net is active, the LoD-thresholded net
            # (v_fill_lod - v_cut_lod) takes precedence; extrapolating the
            # unthresholded mean here would silently discard the thresholding.
            v_net = float(vols["mean_dh"] * area_polygon)

        # σ components, scaled linearly by the area basis
        sig_corr = est.mean_correlated_polygon
        sig_uncorr = est.mean_uncorrelated_polygon
        sig_bias = est.sigma_a_stable

        def _x(v):
            return None if v is None else float(v) * area_used

        sigma_v_corr = _x(sig_corr)
        sigma_v_uncorr = _x(sig_uncorr)
        sigma_v_bias = _x(sig_bias)

        terms = [t for t in (sigma_v_corr, sigma_v_uncorr) if t is not None]
        if include_bias_se and sigma_v_bias is not None:
            terms.append(sigma_v_bias)
        sigma_v = math.sqrt(sum(t ** 2 for t in terms)) if terms else np.nan

        # interval bounds: totals already combine uncorrelated+correlated
        # (and bias if the user asked calc_total_uncertainty to fold it in);
        # rebuild them here from the correlated bounds for consistency with
        # our own include_bias_se handling.
        def _bound(corr_bound):
            if corr_bound is None:
                return None
            parts = [float(corr_bound) * area_used]
            if sigma_v_uncorr is not None:
                parts.append(sigma_v_uncorr)
            if include_bias_se and sigma_v_bias is not None:
                parts.append(sigma_v_bias)
            return math.sqrt(sum(p ** 2 for p in parts))

        sigma_v_min = _bound(est.mean_correlated_polygon_min)
        sigma_v_max = _bound(est.mean_correlated_polygon_max)
        sigma_v_p025 = _bound(est.mean_correlated_polygon_p025)
        sigma_v_p975 = _bound(est.mean_correlated_polygon_p975)

        # pedagogical bounds
        sigma0 = est.sigma0_uncorrelated
        n_valid = vols["n_valid"]
        sigma_v_naive = (
            float(sigma0) * cell_area * math.sqrt(n_valid)
            if sigma0 is not None else np.nan
        )
        sigma_v_fullcorr = (
            float(sigma0) * area_used if sigma0 is not None else np.nan
        )

        unit = getattr(est.raster_data_handler, "unit", "m") or "m"

        result = VolumeResult(
            v_net=v_net,
            v_cut=vols["v_cut"],
            v_fill=vols["v_fill"],
            v_cut_lod=vols["v_cut_lod"],
            v_fill_lod=vols["v_fill_lod"],
            area_polygon=area_polygon,
            area_valid=area_valid,
            area_basis=self.area_basis,
            area_used=area_used,
            frac_valid=vols["frac_valid"],
            n_valid=n_valid,
            cell_area=cell_area,
            mean_dh=vols["mean_dh"],
            sigma_v=sigma_v,
            sigma_v_corr=sigma_v_corr if sigma_v_corr is not None else np.nan,
            sigma_v_uncorr=(sigma_v_uncorr
                            if sigma_v_uncorr is not None else np.nan),
            sigma_v_bias=sigma_v_bias,
            include_bias_se=include_bias_se,
            sigma_v_min=sigma_v_min,
            sigma_v_max=sigma_v_max,
            sigma_v_p025=sigma_v_p025,
            sigma_v_p975=sigma_v_p975,
            sigma_v_naive=sigma_v_naive,
            sigma_v_fullcorr=sigma_v_fullcorr,
            lod=lod_value,
            ci_method=getattr(est, "ci_method_used", None),
            unit=str(unit),
        )
        if threshold_net and lod_value:
            result.extras["v_net_unthresholded"] = vols["v_net"]
        return result
