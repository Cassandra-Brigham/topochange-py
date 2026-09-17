"""Heteroscedastic (spatially-varying) error modelling for topochange.

This module extends the topochange uncertainty framework from a single
stationary error (one scalar σ / total sill, as used by
:class:`~topochange.uncertainty.RegionalUncertaintyEstimator`) to a
*heteroscedastic* model in which the elevation-difference error σ(x) varies
per pixel as a function of terrain and (for lidar) point-cloud predictors,
following Hugonnet et al. (2022, https://doi.org/10.1109/JSTARS.2022.3188922).

The Hugonnet workflow, as realised here on top of the existing topochange
machinery, is:

1. **Predictors.** Terrain predictors (slope, aspect, roughness, curvature)
   come from :class:`topochange.Raster` derivatives; lidar predictors (point
   density, range-normalised intensity, incidence angle, scan angle, GPS
   time / flight-strip id, dominant classification) are aggregated to the
   reference grid from the point cloud via the project's PDAL wrapper
   (:mod:`topochange.pdal_wrapper`); see
   :func:`extract_pointcloud_predictors`.
2. **σ model.** :func:`fit_sigma_model` bins the stable-terrain dh by the
   predictors, computes a robust NMAD per bin, and fits a GAM (``pygam``) to
   ``log(NMAD)``, with categorical predictors (classification, strip id)
   entering as additive log-offsets (or as separate strata). The result is a
   callable :class:`HeteroscedasticSigmaModel` that returns a per-pixel σ.
3. **Standardisation.** ``z = dh / σ(x)`` (:func:`standardize`) yields a
   dimensionless residual whose variogram is a *correlogram* (unit sill).
4. **(Anisotropic) correlogram.** :func:`directional_empirical_variogram`
   estimates a directional variogram of ``z``; :func:`fit_anisotropic_variogram`
   fits a nested :class:`~topochange.composite_variogram.CompositeVariogramModel`
   wrapped by :class:`AnisotropicCompositeVariogram` (geometric anisotropy:
   ratio + axis).
5. **Propagation.** :class:`HeteroscedasticUncertaintyEstimator` propagates to
   a feature of interest using the heteroscedastic Krige relation
   ``C(xᵢ,xⱼ) = σ(xᵢ)·σ(xⱼ)·ρ(hᵢⱼ)``, via either a correlated-field Monte
   Carlo simulation or a double-integral pair estimator.

Design notes
------------
* Heavy / optional dependencies (``scipy``, ``pygam``, ``pdal``, ``rasterio``,
  ``geopandas``, ``shapely``) are imported lazily inside the functions that
  use them, mirroring the rest of the package, so importing this module (and
  ``topochange``) never requires them.
* Robust scale is the NMAD, ``1.4826 · median(|x − median(x)|)``, matching the
  convention used elsewhere in topochange.
* The isotropic special case of the heteroscedastic areal propagation is also
  wired into :class:`~topochange.uncertainty.RegionalUncertaintyEstimator` via
  its ``sigma_field`` argument; this module additionally supports anisotropy.

References
----------
Hugonnet et al. (2022), "Uncertainty analysis of digital elevation models by
spatial inference from stable terrain", IEEE JSTARS 15, 6456-6472.
Rolstad et al. (2009), J. Glaciol. 55(192).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # import only for type checkers; avoids pulling scipy at import time
    from .composite_variogram import CompositeVariogramModel

__all__ = [
    "NMAD_SCALE",
    "nmad",
    "robust_center",
    "ClassificationSpec",
    "read_sbet",
    "read_trajectory_csv",
    "project_trajectory",
    "extract_pointcloud_predictors",
    "build_predictor_frame",
    "predictor_collinearity",
    "HeteroscedasticSigmaModel",
    "fit_sigma_model",
    "standardize",
    "qq_stats",
    "AnisotropicCompositeVariogram",
    "directional_empirical_variogram",
    "fit_anisotropic_variogram",
    "HeteroscedasticUncertaintyEstimator",
    "derive_strip_ids",
    "derive_strip_ids_from_point_source_id",
    "is_point_source_id_informative",
    "run_heteroscedastic_pipeline",
]


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------

NMAD_SCALE = 1.4826  # normal-consistency constant for the MAD


def nmad(x: np.ndarray) -> float:
    """Normalised median absolute deviation, ``1.4826·median(|x−median x|)``.

    Parameters
    ----------
    x : array-like
        Values (NaN / non-finite entries are ignored).

    Returns
    -------
    float
        Robust standard-deviation estimate, or ``nan`` if ``x`` is empty.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return NMAD_SCALE * np.median(np.abs(x - med))


def robust_center(x: np.ndarray) -> float:
    """Median of the finite entries of ``x`` (``nan`` if empty)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


# ---------------------------------------------------------------------------
# Classification handling (ASPRS)
# ---------------------------------------------------------------------------

#: Default ASPRS classification codes. Override via ``ClassificationSpec.codes``.
ASPRS_DEFAULT: Dict[str, int] = {
    "ground": 2,
    "low_veg": 3,
    "med_veg": 4,
    "high_veg": 5,
    "building": 6,
}

#: Integer dominant-class codes used on the raster grid and their labels.
CLASS_INT_TO_LABEL: Dict[int, str] = {0: "other", 1: "ground", 2: "building"}


@dataclass
class ClassificationSpec:
    """Which ASPRS classes are meaningfully populated in the input point cloud.

    The point-filter rule applied at PDAL import (Hugonnet-style, prioritising
    the last-return ground surface under canopy) is:

    ======================  ==================================================
    condition               points kept
    ======================  ==================================================
    ``veg``                 last returns only (``ReturnNumber == NumberOfReturns``)
    else ``building``       all returns
    else ``ground``         last returns (used as a ground proxy)
    else                    all returns (fallback)
    ======================  ==================================================

    Parameters
    ----------
    veg, building, ground : bool
        Whether that class is populated / should drive the point-filter rule
        above.  ``veg`` now affects only point *selection* (last returns
        for ground under canopy); vegetation is no longer a dominant-class
        category; encode canopy as a continuous predictor instead (e.g. a CHM
        passed via ``run_heteroscedastic_pipeline(extra_raster_paths=...)``).
    codes : dict
        Mapping of coarse class name -> ASPRS integer code.
    """

    veg: bool = False
    building: bool = False
    ground: bool = False
    codes: Dict[str, int] = field(default_factory=lambda: dict(ASPRS_DEFAULT))

    def pdal_filter_expression(self) -> Optional[str]:
        """Return a PDAL ``filters.expression`` string, or ``None`` for no filter."""
        if self.veg:
            return "(ReturnNumber == NumberOfReturns)"
        if self.building:
            return None  # keep all returns
        if self.ground:
            return "(ReturnNumber == NumberOfReturns)"
        return None

    def label_for_point(self, class_code: int) -> str:
        """Coarse per-point label used as a GAM factor.

        Returns one of ``'building' | 'ground' | 'other'``.  Vegetation is
        deliberately NOT a category; canopy is expected to enter the σ model as
        a continuous predictor (e.g. a CHM via ``extra_rasters``), so veg returns
        fall through to ``'other'`` here.
        """
        c = self.codes
        if self.building and class_code == c.get("building"):
            return "building"
        if self.ground and class_code == c.get("ground"):
            return "ground"
        return "other"


# ---------------------------------------------------------------------------
# Trajectory handling (optional; enables true beam-vector incidence angles)
# ---------------------------------------------------------------------------

def read_sbet(sbet_path: str) -> pd.DataFrame:
    """Read an Applanix / NovAtel SBET binary trajectory file.

    Each SBET record is 17 float64s (136 bytes): ``time, lat, lon, alt,
    x_vel, y_vel, z_vel, roll, pitch, platform_heading, wander_angle,
    x_accel, y_accel, z_accel, x_ang_rate, y_ang_rate, z_ang_rate``.
    ``lat``/``lon`` are in **radians**; use :func:`project_trajectory` to
    reproject into the raster CRS before use.

    Returns
    -------
    pandas.DataFrame
        Columns ``gps_time, lat_deg, lon_deg, alt, roll, pitch, heading``
        (angles in degrees, ``alt`` in metres).
    """
    raw = np.fromfile(sbet_path, dtype=np.float64)
    if raw.size % 17 != 0:
        raise ValueError(f"SBET size {raw.size * 8} bytes is not a multiple of 136.")
    a = raw.reshape(-1, 17)
    return pd.DataFrame(
        {
            "gps_time": a[:, 0],
            "lat_deg": np.rad2deg(a[:, 1]),
            "lon_deg": np.rad2deg(a[:, 2]),
            "alt": a[:, 3],
            "roll": np.rad2deg(a[:, 7]),
            "pitch": np.rad2deg(a[:, 8]),
            "heading": np.rad2deg(a[:, 9]),
        }
    )


def read_trajectory_csv(csv_path: str) -> pd.DataFrame:
    """Load a generic trajectory CSV.

    Required columns: ``gps_time, x, y, z`` (in the raster CRS, metres).
    Optional attitude columns ``roll, pitch, heading`` (degrees) are retained
    if present but not required; the beam vector is derived from geometry
    (point -> sensor).
    """
    df = pd.read_csv(csv_path)
    required = {"gps_time", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trajectory CSV missing columns: {missing}")
    return df.sort_values("gps_time").reset_index(drop=True)


def project_trajectory(traj: pd.DataFrame, src_crs, dst_crs) -> pd.DataFrame:
    """Reproject a lat/lon SBET frame into ``dst_crs`` (adds ``x, y, z``)."""
    from pyproj import Transformer  # lazy

    tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x, y = tr.transform(traj["lon_deg"].values, traj["lat_deg"].values)
    out = traj.copy()
    out["x"] = x
    out["y"] = y
    out["z"] = traj["alt"].values
    return out


def _interp_trajectory(traj: pd.DataFrame, gps_time: np.ndarray) -> np.ndarray:
    """Linear-interpolate sensor ``(x, y, z)`` at each point's GPS time.

    Returns an ``(n_points, 3)`` array. Points outside the trajectory time
    range are clamped to the nearest endpoint (with a warning).
    """
    t = traj["gps_time"].values
    if not np.all(np.diff(t) >= 0):
        order = np.argsort(t)
        traj = traj.iloc[order]
        t = traj["gps_time"].values
    tmin, tmax = t[0], t[-1]
    n_out = int(((gps_time < tmin) | (gps_time > tmax)).sum())
    if n_out:
        warnings.warn(
            f"{n_out} points fall outside the trajectory time range "
            f"[{tmin:.2f}, {tmax:.2f}]; clamping to endpoints.",
            stacklevel=2,
        )
    gt = np.clip(gps_time, tmin, tmax)
    xs = np.interp(gt, t, traj["x"].values)
    ys = np.interp(gt, t, traj["y"].values)
    zs = np.interp(gt, t, traj["z"].values)
    return np.stack([xs, ys, zs], axis=1)


# ---------------------------------------------------------------------------
# Point-cloud predictor extraction (via topochange.pdal_wrapper)
# ---------------------------------------------------------------------------

def _run_pdal(pipeline_json: dict) -> np.ndarray:
    """Execute a PDAL pipeline via the project wrapper; return the point array.

    Uses :data:`topochange.pdal_wrapper.pdal` (native PDAL or a conda
    subprocess fallback) rather than importing ``pdal`` directly, so the
    behaviour matches the rest of the package.
    """
    from .pdal_wrapper import pdal  # lazy; PdalModule instance

    p = pdal.Pipeline(json.dumps(pipeline_json))
    p.execute()
    arrays = p.arrays
    if not arrays:
        return np.array([])
    return arrays[0]


def _grid_bbox(ref_transform, ref_shape: Tuple[int, int], buffer_px: float = 1.0):
    """Axis-aligned world-coordinate bounding box of the reference grid.

    Built from the four grid corners, so it stays a *safe superset* even for a
    rotated / sheared transform: any point outside this bbox is also outside the
    grid and would be dropped by the per-pixel index test anyway. Returned as
    ``(minx, maxx, miny, maxy)``, expanded by ``buffer_px`` pixels.
    """
    rows, cols = ref_shape
    t = ref_transform
    corners = [(0, 0), (cols, 0), (0, rows), (cols, rows)]
    xs = [t.c + c * t.a + r * t.b for c, r in corners]
    ys = [t.f + c * t.d + r * t.e for c, r in corners]
    px = max(abs(t.a), abs(t.b))
    py = max(abs(t.d), abs(t.e))
    return (min(xs) - buffer_px * px, max(xs) + buffer_px * px,
            min(ys) - buffer_px * py, max(ys) + buffer_px * py)


def _dominant_id_from_key_counts(
    uniq_key: np.ndarray, counts: np.ndarray, n_cells: int
) -> np.ndarray:
    """Reduce ``(cell * 65536 + id, count)`` pairs to a per-cell majority id.

    Shared finalisation step for the per-cell dominant-``PointSourceId`` vote
    used by both the batch and streaming point-cloud predictor paths: the
    streaming path accumulates ``(key, count)`` pairs chunk by chunk (see
    :func:`_extract_pointcloud_predictors_streaming`) and merges them exactly
    once, at the end, through this same routine, so the two paths agree
    numerically. Cells with no observed id get ``-1``.
    """
    out = np.full(n_cells, -1, dtype=np.int64)
    if uniq_key.size == 0:
        return out
    cell = uniq_key // 65536
    pid = uniq_key % 65536
    # Sort so each cell's rows are contiguous (primary key) and, within a
    # cell, ordered by descending count then ascending id; the first row of
    # each run is then the majority id (smaller id breaking exact ties).
    order = np.lexsort((pid, -counts, cell))
    cell_s = cell[order]
    pid_s = pid[order]
    first_in_cell = np.empty(cell_s.shape[0], dtype=bool)
    first_in_cell[0] = True
    first_in_cell[1:] = cell_s[1:] != cell_s[:-1]
    out[cell_s[first_in_cell]] = pid_s[first_in_cell]
    return out


def _dominant_id_per_cell(
    cell_index: np.ndarray, point_id: np.ndarray, n_cells: int
) -> np.ndarray:
    """Per-cell majority vote of a 16-bit point attribute (e.g. ``PointSourceId``).

    For each of the ``n_cells`` grid cells, returns the ``point_id`` value that
    occurs most often among the points assigned to it (ties broken toward the
    smaller id); cells with no points get ``-1``.

    ``point_id`` is assumed to fit in an unsigned 16-bit range, as LAS
    ``PointSourceId`` does by spec, so ``cell_index * 65536 + point_id`` is a
    collision-free ``int64`` key. Aggregating through that key keeps memory
    and time proportional to the number of *distinct* ``(cell, id)`` pairs
    actually observed rather than ``n_cells * n_distinct_ids``; safe even
    when the id range is wide and sparse, unlike a dense per-cell histogram
    (which is what this module uses for the ``dom_class`` vote below, where a
    fixed, tiny category count of 4 makes the simpler dense approach fine).
    """
    if cell_index.size == 0:
        return np.full(n_cells, -1, dtype=np.int64)
    key = cell_index.astype(np.int64) * 65536 + point_id.astype(np.int64)
    uniq_key, counts = np.unique(key, return_counts=True)
    return _dominant_id_from_key_counts(uniq_key, counts, n_cells)


def extract_pointcloud_predictors(
    las_path: str,
    ref_transform,
    ref_shape: Tuple[int, int],
    ref_crs,
    class_spec: ClassificationSpec,
    sensor_altitude: Optional[float] = None,
    normal_from_slope_aspect: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    trajectory: Optional[pd.DataFrame] = None,
    stream: Optional[bool] = None,
    chunk_size: int = 1_000_000,
    crop_to_grid: bool = True,
) -> Dict[str, np.ndarray]:
    """Aggregate LAS/LAZ points to the reference raster grid.

    Parameters
    ----------
    las_path : str
        Path to a LAS/LAZ point cloud (assumed already in ``ref_crs``). The
        cloud may have a *larger extent* than the grid: points outside the grid
        are dropped (and, when ``crop_to_grid``, filtered out by PDAL before
        they ever reach Python).
    ref_transform : affine.Affine
        Reference raster affine transform.
    ref_shape : (int, int)
        Reference raster shape ``(rows, cols)``.
    ref_crs : rasterio CRS
        Reference CRS (unused directly here; assumes points are aligned).
    class_spec : ClassificationSpec
        Class filter rules / codes.
    sensor_altitude : float, optional
        Sensor altitude AGL (m). Used only for intensity range-normalisation
        and the scan-angle range approximation when ``trajectory`` is absent.
    normal_from_slope_aspect : (ndarray, ndarray), optional
        ``(slope_deg, aspect_deg)`` rasters (shape ``ref_shape``) used to build
        the local surface normal for the incidence-angle computation.
    trajectory : pandas.DataFrame, optional
        Columns ``gps_time, x, y, z`` in the raster CRS. If given, the true
        per-point beam vector (point -> sensor) drives the incidence angle;
        otherwise a scan-angle approximation is used.
    stream : bool, optional
        Aggregate the cloud one PDAL chunk at a time (peak memory
        ``O(chunk_size + grid)`` instead of ``O(whole cloud)``) via
        :meth:`PipelineWrapper.iterator`. ``None`` (default) auto-selects
        streaming when the native ``pdal`` module is available, falling back to
        the batch path otherwise; ``True`` forces it (erroring if unavailable);
        ``False`` forces the batch path. Numerically equivalent to the batch
        path except that the global intensity range-normalisation reference
        ``r_ref`` is estimated from a bounded sample of ranges, a global scalar
        that does not affect the scale-invariant σ model (see
        :func:`_extract_pointcloud_predictors_streaming`).
    chunk_size : int
        Points per streamed chunk (streaming path only).
    crop_to_grid : bool
        Prepend a ``filters.crop`` at the grid bounding box so out-of-extent
        points are discarded inside PDAL. Safe (superset) filter (it only drops
        points the per-pixel test would drop anyway) and the main performance
        gain for clouds larger than the difference raster.

    Returns
    -------
    dict of ndarray, each shape ``ref_shape``
        ``density`` (returns/pixel), ``intensity`` (mean, range-normalised if
        possible), ``incidence`` (mean incidence angle, deg), ``scan_angle``
        (mean raw scan angle, deg), ``gps_time_med`` (mean GPS time),
        ``dom_class`` (int code 0=other/1=ground/2=building),
        ``point_source_id`` (int32; the per-cell majority-vote LAS
        ``PointSourceId``, the literal flight-line/strip id assigned by the
        acquisition vendor, present whenever the point cloud populates that
        field; ``-1`` where absent or the cell has no points; see
        :func:`derive_strip_ids_from_point_source_id`), and ``range_mean``
        (mean point->sensor range, only if computable).
    """
    rows, cols = ref_shape

    stages: List[Any] = [las_path]
    if crop_to_grid:
        minx, maxx, miny, maxy = _grid_bbox(ref_transform, ref_shape)
        stages.append({
            "type": "filters.crop",
            "bounds": f"([{minx}, {maxx}], [{miny}, {maxy}])",
        })
    filt = class_spec.pdal_filter_expression()
    if filt:
        stages.append({"type": "filters.expression", "expression": filt})
    pipeline_json = {"pipeline": stages}

    # Streaming vs batch. Streaming keeps peak memory bounded regardless of how
    # much larger the cloud is than the grid, so it is the sensible default when
    # the native module (which alone supports chunked iteration) is present.
    want_stream = stream
    if want_stream is None:
        from .pdal_wrapper import pdal  # lazy
        want_stream = bool(getattr(pdal.Pipeline(json.dumps(pipeline_json)),
                                   "streamable", False))
    if want_stream:
        try:
            return _extract_pointcloud_predictors_streaming(
                pipeline_json, ref_transform, ref_shape, class_spec,
                sensor_altitude, normal_from_slope_aspect, trajectory,
                chunk_size=chunk_size,
            )
        except Exception as e:  # pragma: no cover - depends on install/pipeline
            if stream is True:
                raise
            warnings.warn(
                f"Streaming point-cloud aggregation failed ({e}); falling back "
                f"to the in-memory batch path.",
                stacklevel=2,
            )

    pts = _run_pdal(pipeline_json)

    if pts.size == 0:
        empty = np.full(ref_shape, np.nan, dtype=np.float32)
        return {
            "density": np.zeros(ref_shape, dtype=np.float32),
            "intensity": empty.copy(),
            "incidence": empty.copy(),
            "scan_angle": empty.copy(),
            "gps_time_med": empty.copy(),
            "dom_class": np.zeros(ref_shape, dtype=np.int8),
            "point_source_id": np.full(ref_shape, -1, dtype=np.int32),
        }

    def col(name, default=None):
        return pts[name] if name in pts.dtype.names else default

    X = pts["X"].astype(np.float64)
    Y = pts["Y"].astype(np.float64)
    Z = pts["Z"].astype(np.float64)
    intensity = col("Intensity")
    scan_angle = col("ScanAngleRank")
    gps_time = col("GpsTime")
    cls = col("Classification")
    psid = col("PointSourceId")

    # Map XY -> pixel indices via the inverse affine.
    inv = ~ref_transform
    fcol, frow = inv * (X, Y)
    rr = np.floor(frow).astype(np.int64)
    cc = np.floor(fcol).astype(np.int64)
    valid = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    rr, cc = rr[valid], cc[valid]
    X, Y, Z = X[valid], Y[valid], Z[valid]
    if intensity is not None:
        intensity = intensity[valid]
    if scan_angle is not None:
        scan_angle = scan_angle[valid]
    if gps_time is not None:
        gps_time = gps_time[valid]
    if cls is not None:
        cls = cls[valid]
    if psid is not None:
        psid = psid[valid]

    lin = rr * cols + cc
    n_cells = rows * cols

    density = np.bincount(lin, minlength=n_cells).astype(np.float32)

    def cell_mean(vals):
        if vals is None:
            return np.full(n_cells, np.nan, dtype=np.float32)
        s = np.bincount(lin, weights=vals.astype(np.float64), minlength=n_cells)
        c = np.bincount(lin, minlength=n_cells)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        return m.astype(np.float32)

    # --- per-point range (for intensity normalisation + incidence) ---
    sensor_xyz = None
    if trajectory is not None and gps_time is not None:
        sensor_xyz = _interp_trajectory(trajectory, gps_time)
        r_per_point = np.linalg.norm(
            sensor_xyz - np.stack([X, Y, Z], axis=1), axis=1
        )
    elif sensor_altitude is not None:
        if scan_angle is not None:
            r_per_point = sensor_altitude / np.clip(
                np.cos(np.deg2rad(scan_angle.astype(np.float64))), 0.1, None
            )
        else:
            r_per_point = np.full_like(Z, sensor_altitude)
    else:
        r_per_point = None

    # --- intensity (range-normalised to a reference range if available) ---
    if intensity is not None and r_per_point is not None:
        r_ref = float(np.nanmedian(r_per_point))
        intensity_norm = intensity.astype(np.float64) * (r_per_point / r_ref) ** 2
    else:
        intensity_norm = intensity
    intensity_r = cell_mean(intensity_norm).reshape(rows, cols)

    # --- incidence angle vs local surface normal ---
    if normal_from_slope_aspect is not None:
        slope_deg, aspect_deg = normal_from_slope_aspect
        s_rad = np.deg2rad(np.asarray(slope_deg, dtype=float))
        a_rad = np.deg2rad(np.asarray(aspect_deg, dtype=float))
        nx = (np.sin(s_rad) * np.sin(a_rad)).ravel()
        ny = (np.sin(s_rad) * np.cos(a_rad)).ravel()
        nz = np.cos(s_rad).ravel()
        n_at = np.stack([nx[lin], ny[lin], nz[lin]], axis=1)

        if sensor_xyz is not None:
            b_at = sensor_xyz - np.stack([X, Y, Z], axis=1)
            b_norm = np.linalg.norm(b_at, axis=1, keepdims=True)
            b_at = b_at / np.clip(b_norm, 1e-9, None)
        elif scan_angle is not None:
            # Approximation: assume flight is +X in map coords, so the beam
            # tilts in the +Y (cross-track) direction by scan_angle from nadir.
            # Inaccurate on turns and for strips not flown along +X; provide a
            # trajectory for a rigorous incidence angle.
            sa_rad = np.deg2rad(scan_angle.astype(np.float64))
            b_at = np.stack(
                [np.zeros_like(sa_rad), np.sin(sa_rad), np.cos(sa_rad)], axis=1
            )
        else:
            b_at = None

        if b_at is not None:
            cos_i = np.clip(np.abs(np.einsum("ij,ij->i", n_at, b_at)), 1e-6, 1.0)
            incidence_deg = np.rad2deg(np.arccos(cos_i))
            incidence_r = cell_mean(incidence_deg).reshape(rows, cols)
        else:
            incidence_r = np.full(ref_shape, np.nan, dtype=np.float32)
    else:
        incidence_r = (
            cell_mean(scan_angle).reshape(rows, cols)
            if scan_angle is not None
            else np.full(ref_shape, np.nan, dtype=np.float32)
        )

    scan_angle_r = (
        cell_mean(scan_angle).reshape(rows, cols)
        if scan_angle is not None
        else np.full(ref_shape, np.nan, dtype=np.float32)
    )
    gps_time_r = (
        cell_mean(gps_time).reshape(rows, cols)
        if gps_time is not None
        else np.full(ref_shape, np.nan, dtype=np.float32)
    )

    # --- dominant coarse class per cell ---
    dom_class = np.zeros(n_cells, dtype=np.int8)
    if cls is not None:
        labels = np.zeros(cls.shape[0], dtype=np.int8)
        c = class_spec.codes
        if class_spec.ground:
            labels[cls == c.get("ground", -1)] = 1
        if class_spec.building:
            labels[cls == c.get("building", -1)] = 2
        # Vegetation is intentionally NOT a dominant-class category: canopy is
        # carried by a continuous predictor (e.g. a CHM) instead, so veg returns
        # stay label 0 ("other") here.  (The 4th bincount column is left unused.)
        key = lin.astype(np.int64) * 4 + labels.astype(np.int64)
        counts = np.bincount(key, minlength=n_cells * 4).reshape(n_cells, 4)
        dom_class = counts.argmax(axis=1).astype(np.int8)
        dom_class[counts.sum(axis=1) == 0] = 0

    # --- dominant PointSourceId per cell (literal flight-strip signal) ---
    if psid is not None:
        point_source_id = (
            _dominant_id_per_cell(lin, psid, n_cells)
            .astype(np.int32)
            .reshape(rows, cols)
        )
    else:
        point_source_id = np.full((rows, cols), -1, dtype=np.int32)

    out = {
        "density": density.reshape(rows, cols),
        "intensity": intensity_r,
        "incidence": incidence_r,
        "scan_angle": scan_angle_r,
        "gps_time_med": gps_time_r,
        "dom_class": dom_class.reshape(rows, cols),
        "point_source_id": point_source_id,
    }
    if r_per_point is not None:
        out["range_mean"] = cell_mean(r_per_point).reshape(rows, cols)
    return out


def _extract_pointcloud_predictors_streaming(
    pipeline_json: dict,
    ref_transform,
    ref_shape: Tuple[int, int],
    class_spec: ClassificationSpec,
    sensor_altitude: Optional[float],
    normal_from_slope_aspect: Optional[Tuple[np.ndarray, np.ndarray]],
    trajectory: Optional[pd.DataFrame],
    chunk_size: int = 1_000_000,
    r_ref_cap: int = 2_000_000,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Streaming counterpart of :func:`extract_pointcloud_predictors`.

    Reduces the (already cropped / filtered) point cloud to the reference grid
    one PDAL chunk at a time via :meth:`PipelineWrapper.iterator`, so peak memory
    is ``O(chunk_size + grid)`` rather than ``O(whole cloud)``. Every per-cell
    aggregate here is *additive* and mirrors the batch path's ``bincount`` math
    exactly (the cell-mean denominator is the total return count per cell, as in
    :func:`extract_pointcloud_predictors`).

    The sole non-additive quantity is the intensity range-normalisation
    reference ``r_ref`` (a global median of per-point ranges). It is estimated
    from a bounded sample of up to ``r_ref_cap`` ranges rather than the exact
    median. ``r_ref`` only rescales the *whole* intensity raster by a constant,
    and intensity feeds a GAM smooth term that is invariant to a constant
    rescaling of its axis, so the σ model is unaffected.

    Raises
    ------
    RuntimeError
        If the pipeline is not streamable on this install (caller falls back).
    """
    from .pdal_wrapper import pdal  # lazy

    rows, cols = ref_shape
    n_cells = rows * cols
    inv = ~ref_transform
    rng = np.random.default_rng(seed)

    # Pre-ravel the surface normal once (indexed per chunk by pixel).
    if normal_from_slope_aspect is not None:
        slope_deg, aspect_deg = normal_from_slope_aspect
        s_rad = np.deg2rad(np.asarray(slope_deg, dtype=float))
        a_rad = np.deg2rad(np.asarray(aspect_deg, dtype=float))
        nx_r = (np.sin(s_rad) * np.sin(a_rad)).ravel()
        ny_r = (np.sin(s_rad) * np.cos(a_rad)).ravel()
        nz_r = np.cos(s_rad).ravel()

    # Additive per-cell accumulators.
    acc_count = np.zeros(n_cells, dtype=np.int64)
    acc_int = np.zeros(n_cells, dtype=np.float64)   # Σ intensity numerator
    acc_inc = np.zeros(n_cells, dtype=np.float64)
    acc_scan = np.zeros(n_cells, dtype=np.float64)
    acc_gps = np.zeros(n_cells, dtype=np.float64)
    acc_rng = np.zeros(n_cells, dtype=np.float64)
    acc_class = np.zeros(n_cells * 4, dtype=np.int64)
    r_sample: List[np.ndarray] = []
    r_sample_n = 0
    psid_keys_chunks: List[np.ndarray] = []
    psid_counts_chunks: List[np.ndarray] = []

    fields: Optional[set] = None
    has_intensity = has_scan = has_gps = has_cls = has_psid = False
    r_available = intensity_normalized = incidence_available = False
    saw_points = False

    p = pdal.Pipeline(json.dumps(pipeline_json))
    if not getattr(p, "streamable", False):
        raise RuntimeError("PDAL streaming is not available on this install.")

    for pts in p.iterator(chunk_size=chunk_size):
        if pts.size == 0:
            continue
        if fields is None:
            fields = set(pts.dtype.names)
            has_intensity = "Intensity" in fields
            has_scan = "ScanAngleRank" in fields
            has_gps = "GpsTime" in fields
            has_cls = "Classification" in fields
            has_psid = "PointSourceId" in fields
            r_available = (trajectory is not None and has_gps) or (sensor_altitude is not None)
            intensity_normalized = has_intensity and r_available
            incidence_available = normal_from_slope_aspect is not None and (
                (trajectory is not None and has_gps) or has_scan
            )

        X = pts["X"].astype(np.float64)
        Y = pts["Y"].astype(np.float64)
        Z = pts["Z"].astype(np.float64)
        fcol, frow = inv * (X, Y)
        rr = np.floor(frow).astype(np.int64)
        cc = np.floor(fcol).astype(np.int64)
        valid = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        if not valid.any():
            continue
        saw_points = True
        rr, cc = rr[valid], cc[valid]
        X, Y, Z = X[valid], Y[valid], Z[valid]
        lin = rr * cols + cc

        acc_count += np.bincount(lin, minlength=n_cells)

        scan = pts["ScanAngleRank"][valid].astype(np.float64) if has_scan else None
        gps = pts["GpsTime"][valid].astype(np.float64) if has_gps else None
        inten = pts["Intensity"][valid].astype(np.float64) if has_intensity else None
        cls = pts["Classification"][valid] if has_cls else None
        psid = pts["PointSourceId"][valid] if has_psid else None

        # Per-point range (+ sensor position for the true beam vector).
        sensor_xyz = None
        if trajectory is not None and gps is not None:
            sensor_xyz = _interp_trajectory(trajectory, gps)
            rnge = np.linalg.norm(sensor_xyz - np.stack([X, Y, Z], axis=1), axis=1)
        elif sensor_altitude is not None:
            if scan is not None:
                rnge = sensor_altitude / np.clip(np.cos(np.deg2rad(scan)), 0.1, None)
            else:
                rnge = np.full(X.shape, float(sensor_altitude))
        else:
            rnge = None

        if rnge is not None:
            acc_rng += np.bincount(lin, weights=rnge, minlength=n_cells)
            if r_sample_n < r_ref_cap:  # bounded sample for the global r_ref
                r_fin = rnge[np.isfinite(rnge)]
                if r_fin.size:
                    take = min(r_ref_cap - r_sample_n, r_fin.size)
                    if take < r_fin.size:
                        r_fin = r_fin[rng.permutation(r_fin.size)[:take]]
                    r_sample.append(r_fin)
                    r_sample_n += r_fin.size

        # Intensity numerator; the constant 1/r_ref² is applied at finalize.
        if inten is not None:
            num = inten * rnge ** 2 if intensity_normalized else inten
            acc_int += np.bincount(lin, weights=num, minlength=n_cells)

        # Incidence angle vs local surface normal.
        if incidence_available:
            n_at = np.stack([nx_r[lin], ny_r[lin], nz_r[lin]], axis=1)
            if sensor_xyz is not None:
                b = sensor_xyz - np.stack([X, Y, Z], axis=1)
                b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
            else:  # scan-angle approximation (flight assumed +X in map coords)
                sa = np.deg2rad(scan)
                b = np.stack([np.zeros_like(sa), np.sin(sa), np.cos(sa)], axis=1)
            cos_i = np.clip(np.abs(np.einsum("ij,ij->i", n_at, b)), 1e-6, 1.0)
            acc_inc += np.bincount(
                lin, weights=np.rad2deg(np.arccos(cos_i)), minlength=n_cells
            )

        if scan is not None:
            acc_scan += np.bincount(lin, weights=scan, minlength=n_cells)
        if gps is not None:
            acc_gps += np.bincount(lin, weights=gps, minlength=n_cells)

        if cls is not None:
            labels = np.zeros(cls.shape[0], dtype=np.int64)
            c = class_spec.codes
            if class_spec.ground:
                labels[cls == c.get("ground", -1)] = 1
            if class_spec.building:
                labels[cls == c.get("building", -1)] = 2
            # Vegetation is intentionally NOT a dominant-class category (see
            # extract_pointcloud_predictors): veg returns stay label 0 ("other").
            acc_class += np.bincount(lin * 4 + labels, minlength=n_cells * 4)

        if psid is not None:
            # Reduce this chunk to its distinct (cell, id) pairs right away
            # (see _dominant_id_per_cell) so the running accumulator scales
            # with distinct pairs observed, not with total points streamed.
            key_chunk = lin.astype(np.int64) * 65536 + psid.astype(np.int64)
            uk, uc = np.unique(key_chunk, return_counts=True)
            psid_keys_chunks.append(uk)
            psid_counts_chunks.append(uc)

    if not saw_points:
        empty = np.full(ref_shape, np.nan, dtype=np.float32)
        return {
            "density": np.zeros(ref_shape, dtype=np.float32),
            "intensity": empty.copy(),
            "incidence": empty.copy(),
            "scan_angle": empty.copy(),
            "gps_time_med": empty.copy(),
            "dom_class": np.zeros(ref_shape, dtype=np.int8),
            "point_source_id": np.full(ref_shape, -1, dtype=np.int32),
        }

    # --- finalize (denominator = total return count per cell, as in batch) ---
    def cell_mean(sum_arr):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(acc_count > 0, sum_arr / np.maximum(acc_count, 1), np.nan)
        return m.astype(np.float32).reshape(ref_shape)

    def nan_grid():
        return np.full(ref_shape, np.nan, dtype=np.float32)

    scan_r = cell_mean(acc_scan) if has_scan else nan_grid()
    gps_r = cell_mean(acc_gps) if has_gps else nan_grid()

    if has_intensity:
        # Apply the 1/r_ref² scale in float64 and cast to float32 exactly once,
        # matching the batch path's single cast (a double float32 round-trip
        # here would otherwise show up as a ~1-ULP intensity difference).
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_num = np.where(acc_count > 0, acc_int / np.maximum(acc_count, 1), np.nan)
        if intensity_normalized:
            r_all = np.concatenate(r_sample) if r_sample else np.array([np.nan])
            r_ref = float(np.nanmedian(r_all))
            if np.isfinite(r_ref) and r_ref > 0:
                mean_num = mean_num / r_ref ** 2
        intensity_r = mean_num.astype(np.float32).reshape(ref_shape)
    else:
        intensity_r = nan_grid()

    if normal_from_slope_aspect is not None:
        incidence_r = cell_mean(acc_inc) if incidence_available else nan_grid()
    else:  # batch falls back to the scan-angle mean when no normal is supplied
        incidence_r = scan_r

    if has_cls:
        cls_counts = acc_class.reshape(n_cells, 4)
        dom = cls_counts.argmax(axis=1).astype(np.int8)
        dom[cls_counts.sum(axis=1) == 0] = 0
        dom_class = dom.reshape(ref_shape)
    else:
        dom_class = np.zeros(ref_shape, dtype=np.int8)

    if psid_keys_chunks:
        # Merge the per-chunk distinct-(cell,id) tallies (summing counts for
        # any (cell, id) pair that recurred across chunks) before taking the
        # per-cell majority vote; see _dominant_id_from_key_counts.
        all_keys = np.concatenate(psid_keys_chunks)
        all_counts = np.concatenate(psid_counts_chunks)
        merged_key, inv = np.unique(all_keys, return_inverse=True)
        merged_counts = np.bincount(
            inv, weights=all_counts, minlength=merged_key.size
        ).astype(np.int64)
        point_source_id = (
            _dominant_id_from_key_counts(merged_key, merged_counts, n_cells)
            .astype(np.int32)
            .reshape(ref_shape)
        )
    else:
        point_source_id = np.full(ref_shape, -1, dtype=np.int32)

    out = {
        "density": acc_count.astype(np.float32).reshape(ref_shape),
        "intensity": intensity_r,
        "incidence": incidence_r,
        "scan_angle": scan_r,
        "gps_time_med": gps_r,
        "dom_class": dom_class,
        "point_source_id": point_source_id,
    }
    if r_available:
        out["range_mean"] = cell_mean(acc_rng)
    return out


# ---------------------------------------------------------------------------
# Predictor frame
# ---------------------------------------------------------------------------

def build_predictor_frame(
    dh: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    roughness: np.ndarray,
    pc_predictors: Dict[str, np.ndarray],
    stable_mask: Optional[np.ndarray] = None,
    max_slope_deg: float = 89.0,
    add_pixel_index: bool = False,
    extra_rasters: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Flatten rasters into a per-pixel predictor DataFrame.

    Keeps pixels with a finite ``dh`` and slope ``<= max_slope_deg`` and,
    optionally, inside ``stable_mask``.

    Parameters
    ----------
    dh, slope, aspect, roughness : ndarray
        Terrain / difference rasters, all of the same shape.
    pc_predictors : dict
        Output of :func:`extract_pointcloud_predictors` (may be empty).
    stable_mask : ndarray, optional
        Boolean / 0-1 mask of stable terrain. If given, only stable pixels are
        kept (the σ model is trained on stable terrain).
    max_slope_deg : float
        Slopes above this are excluded (degenerate normals / layover).
    add_pixel_index : bool
        If True, add a ``pixel_index`` column (flattened row-major index) so
        predictions can be scattered back to the full grid.
    extra_rasters : dict of ndarray, optional
        Additional named continuous predictor rasters (each the same shape as
        ``dh``) to add as columns, e.g. ``{"chm": chm_array}`` for a canopy
        height model.  List the names you want the σ model to fit in the
        ``predictors`` argument of :func:`fit_sigma_model`.  Pixels where an
        extra predictor is non-finite are skipped at prediction time.

    Returns
    -------
    pandas.DataFrame
        One row per kept pixel with a ``dom_class`` categorical column.
    """
    def flat(a):
        return np.asarray(a).ravel()

    shape = np.asarray(dh).shape
    n = int(np.prod(shape))

    def pc(name):
        return flat(pc_predictors.get(name, np.full(shape, np.nan, dtype=float)))

    df = pd.DataFrame(
        {
            "dh": flat(dh),
            "slope": flat(slope),
            "aspect": flat(aspect),
            "roughness": flat(roughness),
            "density": pc("density"),
            "intensity": pc("intensity"),
            "incidence": pc("incidence"),
            "scan_angle": pc("scan_angle"),
            "gps_time": flat(
                pc_predictors.get("gps_time_med", np.full(shape, np.nan, dtype=float))
            ),
            "dom_class_i": flat(
                pc_predictors.get("dom_class", np.zeros(shape, dtype=np.int8))
            ),
        }
    )
    df["dom_class"] = (
        df["dom_class_i"].map(CLASS_INT_TO_LABEL).fillna("other").astype("category")
    )
    if extra_rasters:
        for _name, _arr in extra_rasters.items():
            df[_name] = flat(_arr)
    if add_pixel_index:
        df["pixel_index"] = np.arange(n, dtype=np.int64)

    keep = np.isfinite(df["dh"].values) & (df["slope"].values <= max_slope_deg)
    if stable_mask is not None:
        # Nodata pixels in the mask arrive as NaN (see _read_raster), and
        # NaN.astype(bool) is True, which would silently mark every nodata
        # pixel (e.g. a 0/1 mask written with nodata=0) as stable and
        # disable masking entirely.  Treat non-finite and <= 0 as NOT stable.
        m = flat(stable_mask).astype(float)
        keep &= np.isfinite(m) & (m > 0)
    return df.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Collinearity diagnostic
# ---------------------------------------------------------------------------

def predictor_collinearity(
    df: pd.DataFrame,
    predictors: Optional[Sequence[str]] = None,
    *,
    method: str = "spearman",
    high_threshold: float = 0.8,
    compute_vif: bool = True,
    plot: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Collinearity diagnostic for the continuous σ-model predictors.

    Reports a pairwise correlation matrix and per-predictor variance inflation
    factors (VIF) over the complete-case rows of ``df``; use it to decide which
    redundant predictor to drop before :func:`fit_sigma_model`.  Dropping the
    less-physical of a correlated pair keeps the σ model interpretable and
    relieves the joint-bin sparsity of :func:`_bin_predictors`, unlike a PCA
    rotation (which is linear, unsupervised w.r.t. σ, and destroys attribution).

    Parameters
    ----------
    df : pandas.DataFrame
        A predictor frame from :func:`build_predictor_frame` (or the
        ``standardized_frame`` returned by :func:`run_heteroscedastic_pipeline`,
        which retains the predictor columns).
    predictors : sequence of str, optional
        Continuous predictors to assess.  Default: every numeric column except
        the bookkeeping columns (``dh``, ``pixel_index``, ``x``, ``y``,
        ``dom_class``/``dom_class_i``, ``strip_id``, ``gps_time``, ``sigma_hat``,
        ``z``).  Pass the same tuple you give ``fit_sigma_model`` to assess
        exactly the modelled set.
    method : {'spearman', 'pearson', 'kendall'}
        Pairwise correlation method.  Default ``'spearman'`` (rank-based, robust
        to the monotone-nonlinear predictor-error relationships here).
    high_threshold : float
        Absolute correlation at/above which a pair is flagged.
    compute_vif : bool
        Also compute per-predictor VIF (standard *linear* multicollinearity
        diagnostic: ``VIF_j = 1/(1 − R²_j)`` from OLS of predictor j on the
        others).  Rules of thumb: >5 notable, >10 severe.
    plot : bool
        Also return a correlation-heatmap figure under key ``'figure'``.
    verbose : bool
        Print a short summary (VIF table + flagged pairs).

    Returns
    -------
    dict
        ``correlation`` (DataFrame), ``vif`` (Series, descending; if requested),
        ``high_pairs`` (list of ``(a, b, corr)``), ``method``, ``n`` (complete-
        case row count), ``predictors``, and ``figure`` (if ``plot``).

    Notes
    -----
    VIF is a *linear* diagnostic; the GAM's nonlinear analogue is concurvity,
    which ``pygam`` does not expose; a high VIF is a conservative proxy.
    Circular predictors (e.g. ``aspect`` in degrees) are correlated naively
    here; treat their numbers with care.
    """
    EXCLUDE = {
        "dh", "pixel_index", "x", "y", "dom_class", "dom_class_i",
        "strip_id", "gps_time", "sigma_hat", "z",
    }
    if predictors is None:
        predictors = [
            c for c in df.columns
            if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        predictors = [p for p in predictors if p in df.columns]
    predictors = list(dict.fromkeys(predictors))  # de-dup, preserve order
    if len(predictors) < 2:
        raise ValueError(
            f"Need >= 2 numeric predictors present in df; got {predictors}."
        )

    X = df[predictors].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    n = int(len(X))
    if n < max(10, len(predictors) + 2):
        warnings.warn(
            f"Only {n} complete-case rows for {len(predictors)} predictors; "
            "collinearity estimates will be unstable.",
            stacklevel=2,
        )

    # drop zero-variance predictors (correlation / VIF undefined)
    stds = X.std(ddof=0)
    const_cols = [c for c in predictors if float(stds.get(c, 0.0)) == 0.0]
    if const_cols:
        warnings.warn(
            f"Dropping constant predictor(s) {const_cols} (no variance).",
            stacklevel=2,
        )
        predictors = [p for p in predictors if p not in const_cols]
        X = X[predictors]
    if len(predictors) < 2:
        raise ValueError("Fewer than 2 non-constant predictors remain.")

    corr = X.corr(method=method)
    result: Dict[str, Any] = {
        "method": method, "n": n, "predictors": list(predictors),
        "correlation": corr,
    }

    if compute_vif:
        Xv = X.to_numpy(dtype=float)
        mu = Xv.mean(axis=0)
        sd = Xv.std(axis=0, ddof=0)
        Xs = (Xv - mu) / np.where(sd > 0, sd, 1.0)   # standardize (numerical stability)
        vifs: Dict[str, float] = {}
        for j, p in enumerate(predictors):
            y = Xs[:, j]
            A = np.column_stack([np.ones(n), np.delete(Xs, j, axis=1)])
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ beta
            ss_res = float(resid @ resid)
            ss_tot = float(y @ y)                      # standardized -> TSS = n
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2 = min(max(r2, 0.0), 1.0 - 1e-12)
            vifs[p] = 1.0 / (1.0 - r2)
        result["vif"] = pd.Series(vifs, name="VIF").sort_values(ascending=False)

    pairs: List[Tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for k in range(i + 1, len(cols)):
            c = corr.iloc[i, k]
            if np.isfinite(c) and abs(c) >= high_threshold:
                pairs.append((cols[i], cols[k], float(c)))
    pairs.sort(key=lambda t: -abs(t[2]))
    result["high_pairs"] = pairs

    if verbose:
        print(f"predictor_collinearity  (method={method}, n={n} complete-case pixels)")
        if compute_vif:
            print("\nVIF (linear; >5 notable, >10 severe):")
            for p, v in result["vif"].items():
                flag = "   <-- severe" if v > 10 else ("   <- notable" if v > 5 else "")
                print(f"  {p:<14}{v:8.2f}{flag}")
        if pairs:
            print(f"\n|{method} corr| >= {high_threshold}:")
            for a, b, c in pairs:
                print(f"  {a} ~ {b}: {c:+.2f}")
        else:
            print(f"\nNo predictor pair with |{method} corr| >= {high_threshold}.")

    if plot:
        result["figure"] = _plot_correlation_heatmap(corr, method)

    return result


def _plot_correlation_heatmap(corr: pd.DataFrame, method: str):
    """Annotated correlation heatmap for :func:`predictor_collinearity`."""
    import matplotlib.pyplot as plt  # lazy

    labels = list(corr.columns)
    M = corr.to_numpy()
    k = len(labels)
    fig, ax = plt.subplots(figsize=(0.85 * k + 2.0, 0.85 * k + 1.5))
    im = ax.imshow(M, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels)
    for i in range(k):
        for j in range(k):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.55 else "black", fontsize=8,
                )
    ax.set_title(f"{method.capitalize()} correlation — σ-model predictors")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Heteroscedastic σ model (binned NMAD + GAM)
# ---------------------------------------------------------------------------

@dataclass
class HeteroscedasticSigmaModel:
    """Fitted per-pixel σ model: ``σ(x) = exp(GAM(continuous) + factor offsets)``.

    Attributes
    ----------
    predictors : list of str
        Continuous predictors entering the GAM smooth terms.
    factor_cols : list of str
        Categorical predictors entering as additive log-offsets.
    factor_offsets : dict
        ``{column: {level: log_offset}}``.
    gam : object
        Fitted ``pygam.LinearGAM`` on the continuous predictors (or ``None``
        for a stratified model).
    strat_key : str, optional
        The factor stratified on when ``stratified`` is populated.
    stratified : dict
        ``{level: fitted GAM}`` for a per-class stratified model.
    log_target : bool
        Whether the GAM predicts ``log(NMAD)`` (always True here).
    predictor_ranges : dict, optional
        ``{predictor: (min, max)}`` training ranges; predictors are clipped to
        these at prediction time to prevent extrapolation blow-up.
    """

    predictors: List[str]
    factor_cols: List[str]
    factor_offsets: Dict[str, Dict[str, float]]
    gam: Any
    strat_key: Optional[str] = None
    stratified: Dict[str, Any] = field(default_factory=dict)
    log_target: bool = True
    predictor_ranges: Optional[Dict[str, Tuple[float, float]]] = None

    def _clip(self, X: np.ndarray) -> np.ndarray:
        """Clip predictor columns to their training range.

        Prevents the GAM from extrapolating to a σ blow-up outside the observed
        predictor domain, e.g. a CHM whose stable-terrain training range is ~0,
        evaluated at tall-canopy FOI pixels, where an unconstrained log-σ smooth
        can explode to hundreds/thousands of metres. NaNs pass through unchanged.
        """
        if not self.predictor_ranges:
            return X
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
            vals = df[col].astype(str).values
            out = out + np.array([table.get(v, 0.0) for v in vals])
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return per-row σ for a predictor frame (NaN where predictors missing).

        Rows with any non-finite predictor are skipped (σ = NaN) rather than
        passed to the GAM, which raises on NaN/Inf input.  This is the
        documented contract and is essential for lidar predictors, where
        pixels with no returns carry NaN intensity/incidence/gps_time.
        """
        if self.stratified:
            out = np.full(len(df), np.nan)
            for cls, gam in self.stratified.items():
                mask = (df[self.strat_key].astype(str) == cls).values
                if mask.any():
                    Xp = df.loc[mask, self.predictors].to_numpy(dtype=float)
                    ok = np.isfinite(Xp).all(axis=1)
                    if not ok.any():
                        continue
                    Xc = self._clip(Xp)
                    y = np.full(len(Xp), np.nan)
                    y[ok] = gam.predict(Xc[ok])
                    y = y + self._apply_factor_offset(df.loc[mask])
                    out[mask] = np.exp(y) if self.log_target else y
            return out
        Xp = df[self.predictors].to_numpy(dtype=float)
        ok = np.isfinite(Xp).all(axis=1)
        Xc = self._clip(Xp)
        y = np.full(len(Xp), np.nan)
        if ok.any():
            y[ok] = self.gam.predict(Xc[ok])
        y = y + self._apply_factor_offset(df)
        return np.exp(y) if self.log_target else y


def _bin_predictors(
    df: pd.DataFrame,
    predictors: Sequence[str],
    n_bins: int = 8,
    min_count: int = 30,
) -> pd.DataFrame:
    """Quantile-bin each predictor and compute NMAD of ``dh`` per multi-D bin.

    Returns a bin-centre table with columns ``predictors + ['nmad', 'count']``.
    """
    binned = df.copy()
    edges = {}
    for p in predictors:
        q = np.linspace(0, 1, n_bins + 1)
        e = np.unique(np.quantile(binned[p].dropna(), q))
        if len(e) < 3:
            e = np.array([binned[p].min() - 1e-9, binned[p].max() + 1e-9])
        edges[p] = e
        binned[p + "_bin"] = pd.cut(binned[p], bins=e, include_lowest=True, labels=False)

    keys = [p + "_bin" for p in predictors]
    grouped = binned.dropna(subset=keys).groupby(keys, observed=True)
    rows = []
    for key, g in grouped:
        if len(g) < min_count:
            continue
        row = {"count": len(g), "nmad": nmad(g["dh"].values)}
        if not isinstance(key, tuple):
            key = (key,)
        for p, k in zip(predictors, key):
            e = edges[p]
            k = int(k)
            row[p] = 0.5 * (e[k] + e[k + 1])
        rows.append(row)
    return pd.DataFrame(rows)


def fit_sigma_model(
    df: pd.DataFrame,
    predictors: Sequence[str] = (
        "slope", "roughness", "incidence", "density", "intensity",
    ),
    factor_cols: Sequence[str] = ("dom_class", "strip_id"),
    n_bins: int = 8,
    min_count: int = 30,
    stratify_by_class: bool = False,
    strat_key: str = "dom_class",
    n_splines: int = 8,
) -> HeteroscedasticSigmaModel:
    """Fit a GAM on ``log(NMAD)`` over multi-dimensional predictor bins.

    Categorical factors enter as additive log-offsets estimated from per-level
    NMAD ratios (empirical-Bayes shrinkage by count); the GAM then fits the
    continuous shape on factor-adjusted residuals. Set ``stratify_by_class`` to
    fit one GAM per level of ``strat_key`` instead (Hugonnet's forest finding
    motivates stratification for lidar).

    Parameters
    ----------
    df : pandas.DataFrame
        Stable-terrain predictor frame from :func:`build_predictor_frame`.
    predictors : sequence of str
        Continuous predictors (screened for presence / non-NaN).
    factor_cols : sequence of str
        Categorical predictors (screened for presence).
    n_bins, min_count : int
        Binning controls passed to :func:`_bin_predictors`.
    stratify_by_class : bool
        Fit one GAM per ``strat_key`` level.
    strat_key : str
        Stratification factor.
    n_splines : int
        Spline count per smooth term.

    Returns
    -------
    HeteroscedasticSigmaModel
    """
    from pygam import LinearGAM, s  # lazy

    # Work on a positionally-indexed copy so ndarray indexing by group index is
    # always valid, even if the caller passed a .loc[]-filtered subframe.
    df = df.reset_index(drop=True)

    predictors = [p for p in predictors if p in df.columns and df[p].notna().any()]
    if len(predictors) == 0:
        raise ValueError("No usable continuous predictors after NaN screening.")
    factor_cols = [c for c in factor_cols if c in df.columns]

    # When stratifying, the stratification key is captured by the per-stratum
    # fit, so it must NOT also enter as an additive offset; otherwise the
    # class effect is applied twice (once implicitly by the stratum, once by
    # the offset).  Exclude it from the offset factors entirely.
    offset_cols = [
        c for c in factor_cols if not (stratify_by_class and c == strat_key)
    ]

    # Per-level log-offsets for each offset factor (additive in log-σ; shrunk
    # toward zero by count).  Each factor's effect is divided out before the
    # next is estimated so offsets are (approximately) independent.
    factor_offsets: Dict[str, Dict[str, float]] = {}
    residual_dh = df["dh"].values.copy()
    for col in offset_cols:
        base = nmad(residual_dh)
        table: Dict[str, float] = {}
        for lvl, sub in df.groupby(col, observed=True):
            if len(sub) < 30:
                continue
            lvl_nmad = nmad(residual_dh[sub.index.values])
            if lvl_nmad > 0 and base > 0:
                w = len(sub) / (len(sub) + 100.0)
                table[str(lvl)] = float(w * np.log(lvl_nmad / base))
        factor_offsets[col] = table
        scale = np.array(
            [np.exp(table.get(str(v), 0.0)) for v in df[col].astype(str).values]
        )
        residual_dh = residual_dh / np.where(scale > 0, scale, 1.0)

    df_fit = df.copy()
    df_fit["dh"] = residual_dh

    def _fit_one(sub_df: pd.DataFrame):
        table = _bin_predictors(sub_df, predictors, n_bins=n_bins, min_count=min_count)
        table = table[table["nmad"] > 0].dropna()
        if len(table) < len(predictors) * 3:
            warnings.warn(
                f"Only {len(table)} predictor bins survived; the GAM fit will "
                f"be weak. Consider fewer predictors or a larger stable area.",
                stacklevel=2,
            )
        y = np.log(table["nmad"].values)
        X = table[predictors].values
        terms = s(0, n_splines=n_splines)
        for i in range(1, len(predictors)):
            terms = terms + s(i, n_splines=n_splines)
        gam = LinearGAM(terms).fit(X, y, weights=table["count"].values)
        return gam, table

    # Training range per predictor, used to clip predictors at prediction time
    # so the GAM cannot extrapolate to a σ blow-up outside the domain it was
    # actually fit on. This MUST be the range of the bin-centre table the GAM
    # trains on (below), not the raw per-pixel predictor column: quantile bin
    # *edges* are equal-count, but a bin's representative value is the
    # arithmetic midpoint of its edges, so for a right-skewed predictor (e.g.
    # roughness: mostly small, with a long tail) the top bin's centroid can
    # sit well short of the raw column max. Clipping to the raw max then lets
    # `_clip()` pass real pixels through to a region of the spline the GAM
    # never actually saw a training point near, where an unpenalized-enough
    # P-spline can overshoot by orders of magnitude once exponentiated
    # (observed on real lidar data: roughness bin centroids topped out at
    # ~24 against a raw max of ~45, and evaluating the fitted GAM at the raw
    # max produced sigma ~10^4 m from that term alone).
    def _table_ranges(table: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for _p in predictors:
            _col = pd.to_numeric(table[_p], errors="coerce").to_numpy(dtype=float)
            _col = _col[np.isfinite(_col)]
            if _col.size:
                out[_p] = (float(_col.min()), float(_col.max()))
        return out

    if stratify_by_class and strat_key in df_fit.columns:
        strat: Dict[str, Any] = {}
        predictor_ranges: Dict[str, Tuple[float, float]] = {}
        for cls, sub in df_fit.groupby(strat_key, observed=True):
            if len(sub) < 200:
                continue
            try:
                gam_cls, table_cls = _fit_one(sub)
                strat[str(cls)] = gam_cls
                for _p, (_lo, _hi) in _table_ranges(table_cls).items():
                    if _p in predictor_ranges:
                        _plo, _phi = predictor_ranges[_p]
                        predictor_ranges[_p] = (min(_plo, _lo), max(_phi, _hi))
                    else:
                        predictor_ranges[_p] = (_lo, _hi)
            except Exception as e:  # pragma: no cover - depends on data
                warnings.warn(f"GAM fit failed for stratum {cls}: {e}", stacklevel=2)
        if not strat:
            raise RuntimeError("No stratum produced a fitted GAM.")
        return HeteroscedasticSigmaModel(
            predictors=list(predictors),
            factor_cols=list(offset_cols),
            factor_offsets=factor_offsets,
            gam=None,
            strat_key=strat_key,
            stratified=strat,
            log_target=True,
            predictor_ranges=predictor_ranges,
        )

    gam, table = _fit_one(df_fit)
    return HeteroscedasticSigmaModel(
        predictors=list(predictors),
        factor_cols=list(offset_cols),
        factor_offsets=factor_offsets,
        gam=gam,
        log_target=True,
        predictor_ranges=_table_ranges(table),
    )


# ---------------------------------------------------------------------------
# Standardisation + diagnostics
# ---------------------------------------------------------------------------

def standardize(
    df: pd.DataFrame,
    model: HeteroscedasticSigmaModel,
    *,
    sigma_floor: Optional[float] = None,
    sigma_floor_frac: float = 0.05,
    sigma_ceiling: Optional[float] = None,
    sigma_ceiling_frac: float = 20.0,
    center: bool = True,
    rescale: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``sigma_hat`` and ``z = (dh - c)/sigma_hat``.

    Four safeguards:

    - **sigma floor**: the log-space GAM can extrapolate to arbitrarily tiny
      sigma for sparse predictor combinations; dividing by those produces
      z-values of 10^2-10^3 that destroy the residual diagnostics and the
      downstream correlogram fit (observed: excess kurtosis ~1e4, fitted
      sills ~1e15). ``sigma_hat`` is floored at ``sigma_floor`` (default
      ``sigma_floor_frac`` x the median predicted sigma).
    - **sigma ceiling**: symmetrically, the log-space GAM can also *overshoot*
      to an unrealistically large sigma at a sparsely supported edge of a
      predictor's training domain, even with :meth:`HeteroscedasticSigmaModel
      ._clip` engaged (observed on real lidar data: a right-skewed roughness
      predictor whose top quantile-bin centroid undershot the raw column max,
      so pixels between the two were clipped to a value the GAM had no
      training support near, and the spline overshot by orders of magnitude
      once exponentiated). Because the areal Monte-Carlo estimator averages
      ``sigma**2`` over every pixel in a feature, even a handful of such
      pixels can dominate the result. ``sigma_hat`` is capped at
      ``sigma_ceiling`` (default ``sigma_ceiling_frac`` x the median
      predicted sigma; generous relative to the typical <10x dynamic range
      of a well-behaved scene, so it only engages on genuine blow-ups).
    - **robust centering** (``center=True``): subtracts the robust center of
      ``dh`` so the standardized residual has median ~0 even when the input
      difference was not bias-corrected upstream. The center used is stored
      in ``out.attrs['z_center']``.
    - **final rescaling** (``rescale=True``): after the first pass, sigma_hat
      is multiplied by NMAD(z) so that the standardized residual has unit
      robust dispersion: the second step of the two-step standardization of
      Hugonnet et al. (2022, IEEE JSTARS, sect. III-C; as in xdem's
      ``two_step_standardization``). Without it, a globally mis-scaled GAM
      (e.g. when an unobserved driver widens the residual within predictor
      bins) propagates as a squared bias into Var(mean). The applied factor
      is stored in ``out.attrs['sigma_scale']``; values far from 1 indicate
      the sigma model alone did not capture the dispersion.
    """
    sigma = np.asarray(model.predict(df), dtype=float)
    finite_pos = np.isfinite(sigma) & (sigma > 0)
    med = float(np.nanmedian(sigma[finite_pos])) if finite_pos.any() else np.nan
    if sigma_floor is None:
        sigma_floor = max(1e-6, sigma_floor_frac * med) if np.isfinite(med) else 1e-6
    if sigma_ceiling is None:
        sigma_ceiling = (
            sigma_ceiling_frac * med if (np.isfinite(med) and med > 0) else np.inf
        )
    n_ceiling_clipped = int(np.sum(finite_pos & (sigma > sigma_ceiling)))
    sigma = np.where(
        np.isfinite(sigma), np.clip(sigma, sigma_floor, sigma_ceiling), np.nan
    )
    out = df.copy()
    dh = out["dh"].values.astype(float)
    c = robust_center(dh[np.isfinite(dh)]) if center else 0.0
    z = (dh - c) / sigma
    scale = 1.0
    if rescale:
        z_fin = z[np.isfinite(z)]
        if z_fin.size:
            s = float(nmad(z_fin))
            if np.isfinite(s) and s > 0:
                scale = s
                sigma = sigma * scale
                z = (dh - c) / sigma
    out["sigma_hat"] = sigma
    out["z"] = z
    out.attrs["z_center"] = float(c)
    out.attrs["sigma_floor"] = float(sigma_floor)
    out.attrs["sigma_ceiling"] = float(sigma_ceiling)
    out.attrs["n_sigma_ceiling_clipped"] = n_ceiling_clipped
    out.attrs["sigma_scale"] = float(scale)
    return out


def qq_stats(z: np.ndarray) -> Dict[str, float]:
    """Normality diagnostics on standardised residuals ``z``.

    Returns ``n, median, nmad, skew, kurtosis_excess, ks_stat_vs_N01``. A
    near-zero median, unit NMAD, and small excess kurtosis indicate a
    well-standardised, approximately Gaussian residual.
    """
    from scipy import stats  # lazy

    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    return {
        "n": int(z.size),
        "median": float(np.median(z)) if z.size else np.nan,
        "nmad": float(nmad(z)),
        "skew": float(stats.skew(z)) if z.size else np.nan,
        "kurtosis_excess": float(stats.kurtosis(z)) if z.size else np.nan,
        "ks_stat_vs_N01": float(stats.kstest(z, "norm").statistic) if z.size else np.nan,
    }


# ---------------------------------------------------------------------------
# Anisotropic composite variogram (wraps CompositeVariogramModel)
# ---------------------------------------------------------------------------

@dataclass
class AnisotropicCompositeVariogram:
    """Geometric-anisotropy wrapper around a :class:`CompositeVariogramModel`.

    The wrapped composite provides the isotropic nested shape ``γ_iso(h)``
    (nugget + one or more bounded components). Anisotropy rescales the lag by
    direction ``θ`` relative to the anisotropy axis::

        h_eff = hypot(h·cos Δθ / ratio, h·sin Δθ),   Δθ = θ − axis

    so ``ratio > 1`` gives a *longer* effective range **along** the axis (the
    lag component along the axis is compressed, so γ rises more slowly there).
    When ``anisotropy_ratio == 1`` (default) this reduces exactly to the
    isotropic composite. The wrapped model must be stationary so a finite total
    sill and hence a covariance / correlogram exist.

    Attributes
    ----------
    composite : CompositeVariogramModel
        Fitted, stationary composite model (``set_params`` already called).
    anisotropy_ratio : float
        Range multiplier along the axis relative to across it (major/minor),
        ``>= 1`` for the axis to be the long-range direction.
    anisotropy_angle_deg : float
        Long-range (major) axis azimuth in degrees, measured from +X (east),
        in ``[-180, 180]``.
    """

    composite: CompositeVariogramModel
    anisotropy_ratio: float = 1.0
    anisotropy_angle_deg: float = 0.0

    def total_sill(self) -> float:
        """Total sill (bounded sills + nugget) of the wrapped composite."""
        s = self.composite.get_total_sill()
        if s is None:
            raise ValueError(
                "AnisotropicCompositeVariogram requires a stationary composite "
                "(finite total sill); the wrapped model has unbounded components."
            )
        return float(s)

    def _effective_lag(
        self, h: np.ndarray, direction_deg: Optional[np.ndarray] = None
    ) -> np.ndarray:
        h = np.asarray(h, dtype=np.float64)
        if self.anisotropy_ratio == 1.0 or direction_deg is None:
            return h
        dtheta = np.deg2rad(np.asarray(direction_deg, dtype=float) - self.anisotropy_angle_deg)
        return np.hypot(h * np.cos(dtheta) / self.anisotropy_ratio, h * np.sin(dtheta))

    def gamma(
        self, h: np.ndarray, direction_deg: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Semivariance ``γ(h, θ)``."""
        return self.composite(self._effective_lag(h, direction_deg))

    def covariance(
        self, h: np.ndarray, direction_deg: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Covariance ``C(h, θ) = total_sill − γ(h, θ)``."""
        return self.total_sill() - self.gamma(h, direction_deg)

    def correlogram(
        self, h: np.ndarray, direction_deg: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Standardised correlogram ``ρ(h, θ) = C(h, θ) / total_sill`` (ρ(0)=1)."""
        return self.covariance(h, direction_deg) / max(self.total_sill(), 1e-12)


# ---------------------------------------------------------------------------
# Directional empirical variogram of standardised residuals
# ---------------------------------------------------------------------------

def _pair_subsample(
    coords: np.ndarray,
    values: np.ndarray,
    n_pairs: int = 300_000,
    rng=None,
    return_angles: bool = False,
):
    """Random point-pair semivariance samples.

    Returns ``(distances, half_squared_diffs)`` and, if ``return_angles``, the
    pair azimuth in degrees from +X (east), folded to ``[0, 180)``.
    """
    rng = np.random.default_rng(rng)
    n = coords.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    dxy = coords[i] - coords[j]
    d = np.linalg.norm(dxy, axis=1)
    sq = 0.5 * (values[i] - values[j]) ** 2
    if return_angles:
        ang = np.mod(np.rad2deg(np.arctan2(dxy[:, 1], dxy[:, 0])), 180.0)
        return d, sq, ang
    return d, sq


def directional_empirical_variogram(
    df: pd.DataFrame,
    value_col: str = "z",
    n_pairs: int = 300_000,
    n_bins: int = 40,
    max_lag: Optional[float] = None,
    directions: Optional[Sequence[Tuple[float, float]]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Empirical (optionally directional) variogram of a standardised residual.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``x``, ``y`` pixel-centre coordinates and ``value_col``.
    value_col : str
        Column to compute the variogram on (``'z'`` = standardised residual).
    n_pairs, n_bins : int
        Number of random pairs and lag bins.
    max_lag : float, optional
        Maximum lag; defaults to the 60th percentile of sampled distances.
    directions : sequence of (center_deg, tol_deg), optional
        Angular sectors (azimuth from +X, folded to ``[0, 180)``). If ``None``,
        an omnidirectional variogram is returned (``direction = -1``).
    seed : int, optional
        RNG seed.

    Returns
    -------
    pandas.DataFrame
        Columns ``lag, gamma, n, direction``.
    """
    if "x" not in df or "y" not in df:
        raise ValueError("df must contain 'x' and 'y' coordinate columns.")
    coords = df[["x", "y"]].values
    z = df[value_col].values
    finite = np.isfinite(z) & np.isfinite(coords).all(axis=1)
    coords, z = coords[finite], z[finite]
    if coords.shape[0] < 2:
        raise ValueError(
            "Need at least 2 finite observations (with finite coordinates) to "
            f"estimate a variogram; got {coords.shape[0]}."
        )

    want_angles = directions is not None
    rng = np.random.default_rng(seed)
    res = _pair_subsample(coords, z, n_pairs=n_pairs, rng=rng, return_angles=want_angles)
    if want_angles:
        d, sq, ang = res
    else:
        d, sq = res
        ang = None

    if max_lag is None:
        max_lag = float(np.quantile(d, 0.6))
    keep = d <= max_lag
    d, sq = d[keep], sq[keep]
    if ang is not None:
        ang = ang[keep]
    edges = np.linspace(0, max_lag, n_bins + 1)

    def _one_dir(mask, label):
        dd, ss = (d[mask], sq[mask]) if mask is not None else (d, sq)
        idx = np.digitize(dd, edges) - 1
        rows = []
        for b in range(n_bins):
            sel = idx == b
            if int(sel.sum()) < 50:
                continue
            rows.append(
                {
                    "lag": 0.5 * (edges[b] + edges[b + 1]),
                    "gamma": float(np.mean(ss[sel])),
                    "n": int(sel.sum()),
                    "direction": float(label),
                }
            )
        return pd.DataFrame(rows)

    if not want_angles:
        return _one_dir(None, -1.0)

    frames = []
    for center, tol in directions:
        dang = np.abs(((ang - center + 90) % 180) - 90)
        frames.append(_one_dir(dang <= tol, center))
    return pd.concat(frames, ignore_index=True)


def fit_anisotropic_variogram(
    emp: pd.DataFrame,
    component_names: Sequence[str] = ("spherical", "spherical"),
    include_nugget: bool = True,
    anisotropic: bool = False,
    registry=None,
) -> AnisotropicCompositeVariogram:
    """Least-squares fit of a nested composite (+ optional anisotropy).

    Fits a :class:`CompositeVariogramModel` (default: two nested spherical
    components + nugget, the standardised-residual analogue of the classic
    short-/long-range structure) to an empirical variogram of the standardised
    residual. When ``anisotropic`` and ``emp`` contains more than one direction
    (``direction`` column with several values ``>= 0``), a geometric anisotropy
    ratio and axis are fit jointly.

    Parameters
    ----------
    emp : pandas.DataFrame
        Empirical variogram from :func:`directional_empirical_variogram`
        (columns ``lag, gamma, n``; optionally ``direction``).
    component_names : sequence of str
        Bounded component model names for the composite (see
        :data:`topochange.variogram_models.MODEL_REGISTRY`).
    include_nugget : bool
        Include a nugget component.
    anisotropic : bool
        Fit anisotropy if the empirical variogram is directional.
    registry : VariogramModelRegistry, optional
        Model registry (defaults to the package singleton).

    Returns
    -------
    AnisotropicCompositeVariogram
        With ``composite`` parameters set. ``anisotropy_ratio == 1`` for the
        isotropic fit.
    """
    from scipy.optimize import least_squares  # lazy

    from .composite_variogram import CompositeVariogramModel
    from .variogram_models import MODEL_REGISTRY

    if registry is None:
        registry = MODEL_REGISTRY

    h = emp["lag"].values.astype(float)
    g = emp["gamma"].values.astype(float)
    w = np.sqrt(emp["n"].values.astype(float)) if "n" in emp.columns else np.ones_like(h)

    composite = CompositeVariogramModel(
        component_names=list(component_names),
        include_nugget=include_nugget,
        registry=registry,
    )
    # Bounds/guesses must be computed from SORTED UNIQUE lags: a directional
    # empirical frame concatenates per-direction lag sequences, so the raw
    # lag column restarts several times and np.diff() inside the registry's
    # bound helpers goes negative, which silently produced NEGATIVE range
    # lower bounds (observed: fitted range -1199 m "within bounds"). The
    # gamma reference is capped at its 99th percentile so a few outlier bins
    # cannot inflate the sill upper bound.
    h_ref = np.sort(np.unique(h))
    g_ref = np.minimum(g, np.nanpercentile(g, 99.0)) if g.size else g
    lb, ub = composite.bounds(h_ref, g_ref)
    x0 = composite.default_guess(h_ref, g_ref)
    x0 = np.clip(x0, lb, ub)

    has_dirs = (
        anisotropic
        and "direction" in emp.columns
        and (emp["direction"] >= 0).any()
        and emp["direction"].nunique() > 1
    )

    if not has_dirs:
        def resid(params):
            composite.set_params(np.clip(params, lb, ub))
            return (composite(h) - g) * w

        res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=3000)
        composite.set_params(np.clip(res.x, lb, ub))
        return AnisotropicCompositeVariogram(composite=composite)

    directions = emp["direction"].values.astype(float)
    # Extra anisotropy parameters appended: ratio in [0.1, 10], axis in [-180, 180].
    lb_a = list(lb) + [0.1, -180.0]
    ub_a = list(ub) + [10.0, 180.0]
    x0_a = list(x0) + [1.0, 0.0]
    n_base = len(x0)

    def resid_aniso(params):
        base = np.clip(params[:n_base], lb, ub)
        ratio = float(np.clip(params[n_base], 0.1, 10.0))
        axis = float(params[n_base + 1])
        composite.set_params(base)
        model = AnisotropicCompositeVariogram(
            composite=composite, anisotropy_ratio=ratio, anisotropy_angle_deg=axis
        )
        return (model.gamma(h, directions) - g) * w

    res = least_squares(resid_aniso, x0_a, bounds=(lb_a, ub_a), max_nfev=5000)
    composite.set_params(np.clip(res.x[:n_base], lb, ub))
    return AnisotropicCompositeVariogram(
        composite=composite,
        anisotropy_ratio=float(np.clip(res.x[n_base], 0.1, 10.0)),
        anisotropy_angle_deg=float(res.x[n_base + 1]),
    )


# ---------------------------------------------------------------------------
# Per-pixel σ raster
# ---------------------------------------------------------------------------

def evaluate_sigma_raster(
    model: HeteroscedasticSigmaModel,
    predictor_stack_df: pd.DataFrame,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Evaluate ``σ(x)`` over the full grid.

    ``predictor_stack_df`` must carry a ``pixel_index`` column (flattened
    row-major index) so predictions scatter back to ``shape``; build it with
    :func:`build_predictor_frame` using ``add_pixel_index=True`` and no stable
    mask.
    """
    sigma = model.predict(predictor_stack_df)
    out = np.full(shape[0] * shape[1], np.nan, dtype=np.float32)
    idx = predictor_stack_df["pixel_index"].values.astype(np.int64)
    out[idx] = sigma
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# Heteroscedastic areal uncertainty
# ---------------------------------------------------------------------------

class HeteroscedasticUncertaintyEstimator:
    """Propagate a per-pixel σ field + (anisotropic) correlogram to an area.

    Implements the heteroscedastic Krige relation for the error of the mean
    ``dh`` over a feature of interest (FOI)::

        Var(mean dh) = (1/N²) Σᵢ Σⱼ σ(xᵢ)·σ(xⱼ)·ρ(hᵢⱼ)

    where ``ρ`` is the standardised correlogram (unit sill) and ``σ`` is the
    per-pixel field. Two estimators are provided:

    * :meth:`areal_uncertainty_mc`: direct correlated-field Monte Carlo
      simulation (Cholesky of the pixel correlation matrix); returns the NMAD
      of the realised area means and full diagnostics.
    * :meth:`areal_uncertainty_pairwise`: random point-pair estimator of the
      double sum (cheaper, no Cholesky), returning ``sqrt(Var(mean))``.

    Parameters
    ----------
    sigma_raster : ndarray
        Per-pixel σ field aligned to ``transform`` (NaN outside coverage).
    transform : affine.Affine
        Affine transform of ``sigma_raster``.
    variogram : AnisotropicCompositeVariogram
        Fitted correlogram model.
    """

    def __init__(self, sigma_raster: np.ndarray, transform, variogram: AnisotropicCompositeVariogram):
        self.sigma_raster = np.asarray(sigma_raster, dtype=float)
        self.transform = transform
        self.variogram = variogram

    # -- shared helpers ------------------------------------------------------

    def _pixels_in_polygon(self, foi_geom):
        """Return ``(xs, ys, sigma)`` for pixel centres inside ``foi_geom``.

        Containment is evaluated with the vectorised ``shapely.contains_xy``
        (shapely 2.0) rather than a per-pixel Python loop; ~10-100x faster on
        large / multipart polygons, which matters when running Monte Carlo
        across many features.
        """
        import shapely  # lazy; contains_xy is a single vectorised call

        minx, miny, maxx, maxy = foi_geom.bounds
        inv = ~self.transform
        c0, r0 = inv * (minx, maxy)
        c1, r1 = inv * (maxx, miny)
        r_lo = max(int(np.floor(min(r0, r1))), 0)
        c_lo = max(int(np.floor(min(c0, c1))), 0)
        r_hi = min(int(np.ceil(max(r0, r1))), self.sigma_raster.shape[0] - 1)
        c_hi = min(int(np.ceil(max(c0, c1))), self.sigma_raster.shape[1] - 1)
        if r_hi < r_lo or c_hi < c_lo:
            return np.array([]), np.array([]), np.array([])

        rr, cc = np.mgrid[r_lo:r_hi + 1, c_lo:c_hi + 1]
        rr, cc = rr.ravel(), cc.ravel()
        t = self.transform
        xs = t.c + (cc + 0.5) * t.a + (rr + 0.5) * t.b
        ys = t.f + (cc + 0.5) * t.d + (rr + 0.5) * t.e
        inside = np.asarray(shapely.contains_xy(foi_geom, xs, ys), dtype=bool)
        xs, ys, rr, cc = xs[inside], ys[inside], rr[inside], cc[inside]
        sig = self.sigma_raster[rr, cc]
        finite = np.isfinite(sig)
        return xs[finite], ys[finite], sig[finite]

    def _pair_directions(self, dx, dy):
        return np.mod(np.rad2deg(np.arctan2(dy, dx)), 180.0)

    # -- Monte Carlo correlated-field simulation -----------------------------

    def areal_uncertainty_mc(
        self,
        foi_geom,
        n_realizations: int = 500,
        max_pixels: int = 3000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Areal σ via correlated-field Monte Carlo simulation.

        For each realisation: draw ``z ~ N(0, R)`` with ``R`` the correlogram
        correlation matrix among (sub)sampled FOI pixels, scale to
        ``dh = σ·z``, and average over the polygon. The areal σ is the NMAD of
        the realised means. Cholesky cost is ``O(n³)``; pixels above
        ``max_pixels`` are uniformly subsampled.

        Returns
        -------
        dict
            ``sigma_area`` (NMAD of means), ``sigma_area_std``,
            ``mean_of_realized_means``, ``n_pixels``, ``n_realizations``,
            ``cholesky_jitter``; or a ``note`` if the FOI is too small / the
            correlation matrix is not positive-definite.
        """
        rng = np.random.default_rng(seed)
        xs, ys, sig = self._pixels_in_polygon(foi_geom)
        n_full = len(xs)
        if n_full < 50:
            return {"sigma_area": np.nan, "n_pixels": n_full,
                    "note": "too few pixels in FOI"}
        # Diagonal (uncorrelated) part of the double sum uses ALL FOI pixels:
        # Var(mean) = mean(sigma^2)/n_full + (1 - 1/n_full) * E_offdiag.
        # Estimating the variance of a subsample mean instead (the previous
        # behavior) scales the diagonal as 1/m and systematically
        # overestimates sigma_area for FOIs larger than max_pixels.
        mean_sig2_full = float(np.mean(sig ** 2))
        if n_full > max_pixels:
            idx = rng.choice(n_full, size=max_pixels, replace=False)
            xs, ys, sig = xs[idx], ys[idx], sig[idx]
        m = len(xs)

        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        h = np.hypot(dx, dy)
        if self.variogram.anisotropy_ratio != 1.0:
            corr = self.variogram.correlogram(h, self._pair_directions(dx, dy))
        else:
            corr = self.variogram.correlogram(h)
        corr = 0.5 * (corr + corr.T)
        corr[np.diag_indices_from(corr)] = 1.0

        # Off-diagonal expectation from the (sub)sampled pixel set: an
        # unbiased estimate of E_{i != j}[sigma_i sigma_j rho(h_ij)] over the
        # full FOI, since the subsample is uniform without replacement.
        S = (sig[:, None] * sig[None, :]) * corr
        e_offdiag = float((S.sum() - np.trace(S)) / (m * (m - 1)))
        var_mean = mean_sig2_full / n_full + (1.0 - 1.0 / n_full) * e_offdiag

        jitter = 1e-6
        L = None
        for _ in range(8):
            try:
                L = np.linalg.cholesky(corr + jitter * np.eye(m))
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        if L is None:
            return {
                "sigma_area": float(np.sqrt(max(var_mean, 0.0))),
                "n_pixels": n_full,
                "n_subsample": m,
                "note": ("correlation matrix not positive-definite; "
                         "sigma_area from the pair-expectation formula only"),
            }

        # Realizations retained as a Monte-Carlo diagnostic of the
        # *subsample* mean (NMAD-robust); the reported sigma_area is the
        # corrected full-FOI value above.
        Z = rng.standard_normal(size=(m, n_realizations))
        dh_real = sig[:, None] * (L @ Z)
        means = dh_real.mean(axis=0)
        return {
            "sigma_area": float(np.sqrt(max(var_mean, 0.0))),
            "sigma_area_subsample_nmad": float(nmad(means)),
            "sigma_area_std": float(np.std(means)),
            "mean_of_realized_means": float(np.mean(means)),
            "n_pixels": n_full,
            "n_subsample": m,
            "n_realizations": n_realizations,
            "cholesky_jitter": jitter,
        }

    # -- double-integral pair estimator --------------------------------------

    def areal_uncertainty_pairwise(
        self,
        foi_geom,
        n_pairs: int = 200_000,
        max_pixels: int = 20000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Areal σ via a random point-pair estimator of the double sum.

        ``Var(mean dh) ≈ ⟨σ(xᵢ)·σ(xⱼ)·ρ(hᵢⱼ)⟩`` over random FOI pixel pairs.
        Cheaper than :meth:`areal_uncertainty_mc` (no Cholesky) and suitable
        for large FOIs.

        Returns
        -------
        dict
            ``sigma_area`` = ``sqrt(max(Var, 0))``, ``n_pixels``, ``n_pairs``,
            ``mean_pairwise_covariance``.
        """
        rng = np.random.default_rng(seed)
        xs, ys, sig = self._pixels_in_polygon(foi_geom)
        n_full = len(xs)
        if n_full < 50:
            return {"sigma_area": np.nan, "n_pixels": n_full,
                    "note": "too few pixels in FOI"}
        # Diagonal (i = j) part of the double sum uses ALL FOI pixels; the
        # off-diagonal expectation is estimated from sampled DISTINCT pairs:
        # Var(mean) = mean(sigma^2)/n_full + (1 - 1/n_full) * E_offdiag.
        # (Previously self-pairs entered at weight 1/m of the subsample,
        # overstating the uncorrelated term for FOIs larger than max_pixels.)
        mean_sig2_full = float(np.mean(sig ** 2))
        if n_full > max_pixels:
            idx = rng.choice(n_full, size=max_pixels, replace=False)
            xs, ys, sig = xs[idx], ys[idx], sig[idx]
        m = len(xs)

        ii = rng.integers(0, m, size=n_pairs)
        jj = rng.integers(0, m, size=n_pairs)
        distinct = ii != jj
        ii, jj = ii[distinct], jj[distinct]
        dx = xs[ii] - xs[jj]
        dy = ys[ii] - ys[jj]
        h = np.hypot(dx, dy)
        if self.variogram.anisotropy_ratio != 1.0:
            rho = self.variogram.correlogram(h, self._pair_directions(dx, dy))
        else:
            rho = self.variogram.correlogram(h)
        cov_ij = sig[ii] * sig[jj] * rho
        e_offdiag = float(np.mean(cov_ij))
        var_mean = mean_sig2_full / n_full + (1.0 - 1.0 / n_full) * e_offdiag
        return {
            "sigma_area": float(np.sqrt(max(var_mean, 0.0))),
            "n_pixels": n_full,
            "n_subsample": m,
            "n_pairs": int(distinct.sum()),
            "mean_pairwise_covariance": e_offdiag,
        }


# ---------------------------------------------------------------------------
# Flight-strip id from GPS-time jumps
# ---------------------------------------------------------------------------

def derive_strip_ids(gps_time: np.ndarray, gap_seconds: float = 5.0) -> np.ndarray:
    """Assign a flight-strip id to each pixel from its (median) GPS time.

    Strips are detected as gaps in the sorted GPS-time distribution larger
    than ``gap_seconds``; each contiguous block gets a distinct integer id.
    Pixels with non-finite GPS time get ``-1``. Within a strip trajectory
    errors are correlated; across strips they are essentially independent, so
    ``strip_id`` is a useful categorical σ predictor.
    """
    t = np.asarray(gps_time, dtype=np.float64)
    finite = np.isfinite(t)
    out = np.full(t.shape, -1, dtype=np.int64)
    if not finite.any():
        return out
    t_valid = t[finite]
    order = np.argsort(t_valid)
    t_sorted = t_valid[order]
    breaks = np.where(np.diff(t_sorted) > gap_seconds)[0]
    strip_edges = np.concatenate(
        [
            [t_sorted[0] - 1e-6],
            (t_sorted[breaks] + t_sorted[breaks + 1]) / 2,
            [t_sorted[-1] + 1e-6],
        ]
    )
    ids_sorted = np.digitize(t_sorted, strip_edges) - 1
    ids_valid = np.empty_like(ids_sorted)
    ids_valid[order] = ids_sorted
    out[finite] = ids_valid
    return out


def derive_strip_ids_from_point_source_id(point_source_id: np.ndarray) -> np.ndarray:
    """Assign a flight-strip id to each pixel from its LAS ``PointSourceId``.

    Unlike :func:`derive_strip_ids` (which *infers* strip boundaries from
    jumps in GPS time: a heuristic, since a real gap in acquisition timing
    need not coincide with a new physical strip, and vice versa),
    ``PointSourceId`` is the field the LAS/LAZ specification reserves for the
    flight-line/strip identifier the acquisition vendor assigned at capture
    time. When it is populated (see :func:`extract_pointcloud_predictors`,
    which surfaces the per-cell majority-vote id as ``point_source_id``),
    using it directly is a more literal (and typically more reliable)
    strip signal than the GPS-time heuristic.

    Parameters
    ----------
    point_source_id : ndarray
        Per-pixel dominant raw ``PointSourceId``, e.g. the ``point_source_id``
        array returned by :func:`extract_pointcloud_predictors` /
        :func:`run_heteroscedastic_pipeline`'s point-cloud extraction.
        ``-1`` marks pixels with no point-cloud coverage (or files that don't
        populate the field).

    Returns
    -------
    ndarray of int64, same shape
        Compact sequential strip labels ``0, 1, 2, ...`` (the raw
        ``PointSourceId`` values themselves are not necessarily contiguous,
        e.g. ``{4540, 4541, 4542}``, so they are relabelled in ascending
        order for consistency with :func:`derive_strip_ids`'s contract).
        Pixels with ``point_source_id == -1`` get ``-1`` here too.

    See Also
    --------
    derive_strip_ids : GPS-time-gap heuristic fallback when PointSourceId is
        absent or degenerate (e.g. a vendor left it all-zero).
    """
    psid = np.asarray(point_source_id, dtype=np.int64)
    out = np.full(psid.shape, -1, dtype=np.int64)
    valid = psid >= 0
    if not valid.any():
        return out
    _, inverse = np.unique(psid[valid], return_inverse=True)
    out[valid] = inverse
    return out


def is_point_source_id_informative(point_source_id: np.ndarray, min_strips: int = 2) -> bool:
    """Whether a ``point_source_id`` array carries usable flight-strip signal.

    Some vendors leave LAS ``PointSourceId`` at a single constant value (most
    often ``0``) for an entire delivery, which is technically "populated" but
    carries no information. :func:`run_heteroscedastic_pipeline` uses this
    check to decide, under ``strip_id_source="auto"``, whether to trust
    ``PointSourceId`` or fall back to :func:`derive_strip_ids`.

    Parameters
    ----------
    point_source_id : ndarray
        As returned by :func:`extract_pointcloud_predictors` (``-1`` = no
        coverage).
    min_strips : int
        Minimum number of distinct non-negative ids required to call the
        field informative. Default 2 (a single strip covering the whole grid
        is technically valid but gives ``strip_id`` no explanatory power as a
        *categorical* predictor).

    Returns
    -------
    bool
    """
    psid = np.asarray(point_source_id)
    valid_vals = psid[psid >= 0]
    if valid_vals.size == 0:
        return False
    return int(np.unique(valid_vals).size) >= min_strips


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def _read_raster(path: str):
    """Return ``(array float64, transform, crs, shape)`` for a single-band raster."""
    import rasterio  # lazy

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        return arr, src.transform, src.crs, arr.shape


def _as_path_list(paths) -> List[str]:
    """Normalize a path / sequence-of-paths argument to a list of paths.

    A bare ``str`` / ``bytes`` / ``os.PathLike`` is wrapped in a single-element
    list (so it is not iterated character-by-character, which would try to open
    ``'/'`` as a raster); ``None`` -> ``[]``.
    """
    if paths is None:
        return []
    if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__"):
        return [paths]
    return list(paths)


def _reproject_to_grid(
    src_arr, src_transform, src_crs, dst_transform, dst_crs, dst_shape,
    resampling: str = "bilinear",
):
    """Reproject/resample an in-memory single-band array onto a target grid.

    Returns ``(array float64 on the dst grid, was_resampled: bool)`` with nodata
    carried as NaN.  If the source already sits on the target grid (same CRS,
    transform, shape) the array is returned unchanged (fast path: no
    interpolation).  ``resampling`` is a ``rasterio.warp.Resampling`` name
    (``'bilinear'`` for continuous predictors, ``'nearest'`` for masks/validity).
    """
    from rasterio.warp import reproject, Resampling  # lazy

    src_arr = np.asarray(src_arr, dtype=np.float64)
    if src_crs is None:
        src_crs = dst_crs  # source without a CRS: assume it matches the dh grid
    if (src_crs == dst_crs and src_transform == dst_transform
            and src_arr.shape == tuple(dst_shape)):
        return src_arr, False
    dst = np.full(tuple(dst_shape), np.nan, dtype=np.float64)
    reproject(
        source=src_arr, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=getattr(Resampling, resampling),
    )
    return dst, True


def _read_raster_on_grid(path, dst_transform, dst_crs, dst_shape, resampling="bilinear"):
    """Read a single-band raster and reproject/resample it onto a target grid.

    Returns ``(array float64 on the dst grid, was_resampled: bool)`` with nodata
    as NaN (see :func:`_reproject_to_grid`).
    """
    import rasterio  # lazy

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        src_transform, src_crs = src.transform, src.crs
    return _reproject_to_grid(
        arr, src_transform, src_crs, dst_transform, dst_crs, dst_shape, resampling
    )


def run_heteroscedastic_pipeline(
    dh_path: str,
    las_path: Optional[str] = None,
    slope_path: Optional[str] = None,
    aspect_path: Optional[str] = None,
    roughness_path: Optional[str] = None,
    reference_dem_path: Optional[str] = None,
    class_spec: Optional[ClassificationSpec] = None,
    stable_mask_path: Optional[str] = None,
    stable_area_paths: Optional[Sequence[str]] = None,
    extra_raster_paths: Optional[Dict[str, str]] = None,
    extra_raster_fill: Optional[Dict[str, float]] = None,
    resampling: str = "bilinear",
    foi_geometries: Optional[Dict[str, Any]] = None,
    sensor_altitude: Optional[float] = None,
    sbet_path: Optional[str] = None,
    trajectory_csv: Optional[str] = None,
    trajectory_crs: Optional[Any] = None,
    stream_pointcloud: Optional[bool] = None,
    pc_chunk_size: int = 1_000_000,
    predictors: Sequence[str] = ("slope", "roughness", "incidence", "density", "intensity"),
    factor_cols: Sequence[str] = ("dom_class", "strip_id"),
    sigma_n_bins: int = 8,
    sigma_min_count: int = 30,
    stratify_by_class: bool = False,
    strip_time_gap_s: float = 5.0,
    include_strip_id: bool = True,
    strip_id_source: str = "auto",
    variogram_directions: Optional[Sequence[Tuple[float, float]]] = None,
    variogram_components: Sequence[str] = ("spherical", "spherical"),
    variogram_n_pairs: int = 300_000,
    variogram_n_bins: int = 40,
    fit_anisotropy: bool = False,
    areal_method: str = "mc",
    mc_realizations: int = 500,
    n_pairs: int = 200_000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the full heteroscedastic pipeline on a difference raster.

    Wires together the topochange building blocks: terrain predictors from
    :class:`topochange.Raster` derivatives (or supplied paths), lidar
    predictors from the point cloud, a GAM σ model, a standardised
    (anisotropic) correlogram, and heteroscedastic areal propagation.

    Parameters
    ----------
    dh_path : str
        Bias-adjusted difference raster (median removed) in a projected CRS.
    las_path : str, optional
        LAS/LAZ point cloud aligned to ``dh_path``. If omitted, only terrain
        predictors are used.
    slope_path, aspect_path, roughness_path : str, optional
        Precomputed terrain derivatives. Any missing one is derived from
        ``reference_dem_path`` via :class:`topochange.Raster` if provided.
    reference_dem_path : str, optional
        DEM from which to derive missing slope/aspect/roughness.
    class_spec : ClassificationSpec, optional
        Point-cloud class rules (defaults to all-returns, no filter).
    stable_mask_path : str, optional
        0/1 stable-terrain mask (strongly recommended: the σ model trains on it).
    stable_area_paths : sequence of str, optional
        One or more stable-area rasters whose *valid* (non-nodata) pixels define
        stable terrain, e.g. the outputs of
        :class:`topochange.stable_area_analysis.StableAreaRasterizer`.  The mask
        is the union of finite pixels across the files; if ``stable_mask_path``
        is also given, the two are unioned.  Resampled onto the ``dh_path`` grid
        automatically if needed (nearest).
    extra_raster_paths : dict, optional
        ``{name: path}`` of additional continuous predictor rasters, e.g.
        ``{"chm": "chm.tif"}`` for a canopy height model.  Resampled onto the
        ``dh_path`` grid automatically if needed (see ``resampling``).  Add the
        same name(s) to ``predictors`` for them to enter the σ GAM.
    extra_raster_fill : dict, optional
        ``{name: value}`` to fill non-finite pixels of an extra raster before
        use, e.g. ``{"chm": 0.0}`` so bare-ground (no-canopy) pixels get a σ
        instead of being skipped.  Unfilled non-finite predictors give σ = NaN at
        those pixels.
    resampling : str
        ``rasterio.warp.Resampling`` method used to align the *continuous*
        auxiliary rasters (terrain slope/aspect/roughness and any
        ``extra_raster_paths``) onto the ``dh_path`` grid when they differ.
        Default ``'bilinear'``.  Masks / stable-area validity always use
        ``'nearest'``.
    foi_geometries : dict, optional
        ``{name: shapely geometry}`` features of interest for areal σ.
    sensor_altitude : float, optional
        Sensor altitude AGL (m) for intensity normalisation without a trajectory.
    sbet_path, trajectory_csv, trajectory_crs : optional
        Trajectory inputs for true beam-vector incidence angles.
    stream_pointcloud : bool, optional
        Aggregate the point cloud in bounded-memory streaming chunks
        (``None`` = auto-select when native PDAL is present). Recommended when
        the cloud has a larger extent than ``dh_path``. See
        :func:`extract_pointcloud_predictors`.
    pc_chunk_size : int
        Points per streamed chunk.
    predictors : sequence of str
        Continuous σ predictors.
    factor_cols : sequence of str
        Categorical σ predictors entering as additive log-offsets (or, for the
        stratification key, absorbed by ``stratify_by_class``).
    sigma_n_bins, sigma_min_count : int
        Predictor-binning controls for the σ GAM.
    stratify_by_class : bool
        Fit one σ GAM per dominant class.
    strip_time_gap_s, include_strip_id : float, bool
        Flight-strip detection controls (``strip_time_gap_s`` only matters for
        the ``derive_strip_ids`` GPS-time-gap fallback; see
        ``strip_id_source``).
    strip_id_source : {'auto', 'point_source_id', 'gps_gap'}
        How to derive ``strip_id``. ``'point_source_id'`` uses the point
        cloud's LAS ``PointSourceId`` field directly (see
        :func:`derive_strip_ids_from_point_source_id`), the literal
        vendor-assigned flight-line id, when present. ``'gps_gap'`` always
        uses the :func:`derive_strip_ids` GPS-time-jump heuristic. ``'auto'``
        (default) prefers ``PointSourceId`` when the point cloud populates it
        with more than one distinct value (see
        :func:`is_point_source_id_informative`) and falls back to the
        GPS-time heuristic otherwise, e.g. when ``las_path`` has no
        ``PointSourceId`` dimension, or a vendor left it at a single constant
        value. The source actually used is recorded in
        ``diagnostics['strip_id_source_used']``.
    variogram_directions : sequence of (center, tol), optional
        Directional sectors for the correlogram (required for anisotropy).
    variogram_components : sequence of str
        Composite component names for the correlogram fit.
    variogram_n_pairs, variogram_n_bins : int
        Random-pair count and lag-bin count for the empirical correlogram.
    fit_anisotropy : bool
        Fit a geometric anisotropy ratio + axis.
    areal_method : {'mc', 'pairwise'}
        Areal σ estimator.
    mc_realizations, n_pairs : int
        Estimator controls.
    seed : int, optional
        RNG seed.

    Returns
    -------
    dict
        ``sigma_model``, ``sigma_raster``, ``empirical_variogram``,
        ``variogram_model`` (:class:`AnisotropicCompositeVariogram`),
        ``diagnostics``, ``transform``, ``crs``, and (if ``foi_geometries``)
        ``foi_uncertainty`` ``{name: dict}``.
    """
    if class_spec is None:
        class_spec = ClassificationSpec()

    dh, transform, crs, shape = _read_raster(dh_path)

    # --- terrain predictors: supplied paths or derived from reference DEM ---
    def _terrain(path, kind):
        if path is not None:
            arr, resampled = _read_raster_on_grid(path, transform, crs, shape, resampling)
            if resampled:
                warnings.warn(
                    f"Terrain raster for '{kind}' was resampled onto the dh grid "
                    f"({resampling}).", stacklevel=2,
                )
            return arr
        if reference_dem_path is not None:
            import rasterio  # lazy
            from .raster import Raster  # lazy

            dem = Raster.from_file(reference_dem_path)
            deriv = getattr(dem, kind)()
            arr = np.asarray(deriv.data, dtype=np.float64)
            if arr.ndim == 3:
                arr = arr[0]
            with rasterio.open(reference_dem_path) as _s:
                dem_transform, dem_crs = _s.transform, _s.crs
            arr, resampled = _reproject_to_grid(
                arr, dem_transform, dem_crs, transform, crs, shape, resampling
            )
            if resampled:
                warnings.warn(
                    f"'{kind}' derived from the reference DEM was resampled onto "
                    f"the dh grid ({resampling}).", stacklevel=2,
                )
            return arr
        return np.full(shape, np.nan, dtype=np.float64)

    slope = _terrain(slope_path, "slope")
    aspect = _terrain(aspect_path, "aspect")
    roughness = _terrain(roughness_path, "roughness")

    # --- stable-terrain mask: an explicit 0/1 mask and/or stable-area TIFFs
    # whose valid (non-nodata) pixels are stable (union of both if both given) ---
    stable = None
    if stable_mask_path is not None:
        stable, _sm_res = _read_raster_on_grid(
            stable_mask_path, transform, crs, shape, "nearest"
        )
        if _sm_res:
            warnings.warn(
                "Stable mask was resampled onto the dh grid (nearest).",
                stacklevel=2,
            )
    stable_area_list = _as_path_list(stable_area_paths)
    if stable_area_list:
        stable_valid = np.zeros(shape, dtype=bool)
        for _sp in stable_area_list:
            _arr, _sa_res = _read_raster_on_grid(_sp, transform, crs, shape, "nearest")
            if _sa_res:
                warnings.warn(
                    f"stable_area raster {_sp!r} was resampled onto the dh grid "
                    f"(nearest).", stacklevel=2,
                )
            stable_valid |= np.isfinite(_arr)
        if stable is not None:
            s0 = np.asarray(stable, dtype=float)
            stable = (np.isfinite(s0) & (s0 > 0)) | stable_valid
        else:
            stable = stable_valid

    # --- extra continuous predictor rasters (e.g. a CHM), aligned to dh ---
    extra: Dict[str, np.ndarray] = {}
    if extra_raster_paths:
        for _name, _path in extra_raster_paths.items():
            _arr, _ex_res = _read_raster_on_grid(_path, transform, crs, shape, resampling)
            if _ex_res:
                warnings.warn(
                    f"extra_raster {_name!r} was resampled onto the dh grid "
                    f"({resampling}).", stacklevel=2,
                )
            if extra_raster_fill and _name in extra_raster_fill:
                _arr = np.where(np.isfinite(_arr), _arr, float(extra_raster_fill[_name]))
            extra[_name] = _arr

    # --- trajectory (optional) ---
    trajectory = None
    if sbet_path is not None:
        traj_llh = read_sbet(sbet_path)
        trajectory = project_trajectory(traj_llh, trajectory_crs or "EPSG:4326", crs)
    elif trajectory_csv is not None:
        trajectory = read_trajectory_csv(trajectory_csv)

    # --- point-cloud predictors (optional) ---
    if las_path is not None:
        pc = extract_pointcloud_predictors(
            las_path, transform, shape, crs,
            class_spec=class_spec,
            sensor_altitude=sensor_altitude,
            normal_from_slope_aspect=(slope, aspect),
            trajectory=trajectory,
            stream=stream_pointcloud,
            chunk_size=pc_chunk_size,
        )
    else:
        pc = {}

    # --- flight-strip ids: derive ONCE over the full grid so the same physical
    # strip carries the same integer label in the training and inference frames
    # (deriving per-frame could relabel a strip and silently drop its offset). --
    strip_all = None
    strip_id_source_used = "none"
    if include_strip_id:
        if strip_id_source not in ("auto", "point_source_id", "gps_gap"):
            raise ValueError(
                f"strip_id_source must be 'auto', 'point_source_id', or "
                f"'gps_gap'; got {strip_id_source!r}."
            )
        psid_all = pc.get("point_source_id")
        use_psid = (
            strip_id_source in ("auto", "point_source_id")
            and psid_all is not None
            and is_point_source_id_informative(psid_all)
        )
        if strip_id_source == "point_source_id" and not use_psid:
            warnings.warn(
                "strip_id_source='point_source_id' was requested but the "
                "point cloud's PointSourceId field is absent or carries a "
                "single constant value (uninformative); falling back to the "
                "GPS-time-gap heuristic (derive_strip_ids).",
                stacklevel=2,
            )
        if use_psid:
            strip_all = derive_strip_ids_from_point_source_id(
                np.asarray(psid_all).ravel()
            )
            strip_id_source_used = "point_source_id"
        else:
            gps_all = pc.get("gps_time_med")
            if gps_all is not None and np.isfinite(np.asarray(gps_all)).any():
                strip_all = derive_strip_ids(
                    np.asarray(gps_all).ravel(), gap_seconds=strip_time_gap_s
                )
                strip_id_source_used = "gps_gap"

    def _attach_strip(frame: pd.DataFrame) -> pd.DataFrame:
        if strip_all is not None:
            idx = frame["pixel_index"].values.astype(np.int64)
            frame["strip_id"] = pd.Categorical(strip_all[idx])
        return frame

    # --- stable-terrain frame (train σ) ---
    df = _attach_strip(build_predictor_frame(
        dh, slope, aspect, roughness, pc, stable_mask=stable, add_pixel_index=True,
        extra_rasters=extra,
    ))

    # attach pixel-centre coordinates from pixel_index
    pj = df["pixel_index"].values
    r_idx, c_idx = np.divmod(pj, shape[1])
    df["x"] = transform.c + (c_idx + 0.5) * transform.a + (r_idx + 0.5) * transform.b
    df["y"] = transform.f + (c_idx + 0.5) * transform.d + (r_idx + 0.5) * transform.e

    model = fit_sigma_model(
        df, predictors=predictors, factor_cols=factor_cols,
        n_bins=sigma_n_bins, min_count=sigma_min_count,
        stratify_by_class=stratify_by_class,
    )
    std = standardize(df, model)
    diagnostics = qq_stats(std["z"].values)  # UNclipped: honest diagnostics
    # two-step standardization receipt: nmad(z) = 1 holds by construction
    # after the final rescale, so surface the applied factor; values far
    # from 1 mean the GAM alone did not capture the residual dispersion
    diagnostics["sigma_scale"] = float(std.attrs.get("sigma_scale", 1.0))
    # floor/ceiling actually applied (see `standardize`), surfaced so callers
    # can tell whether the ceiling safeguard engaged at all for this fit
    diagnostics["sigma_floor"] = float(std.attrs.get("sigma_floor", 0.0))
    diagnostics["sigma_ceiling"] = float(std.attrs.get("sigma_ceiling", np.inf))
    # which strip_id source actually got used (see strip_id_source docstring)
    diagnostics["strip_id_source_used"] = strip_id_source_used

    # The correlogram is fit on clipped z: a handful of extreme residuals
    # (data blunders, unmodeled water/edges) otherwise dominate the classical
    # gamma estimator and wreck the fit. Clipping at |z| = z_clip affects
    # only the correlation-structure estimate, not the diagnostics above.
    z_clip = 10.0
    std_v = std.copy()
    std_v["z"] = np.clip(std_v["z"].values, -z_clip, z_clip)
    emp = directional_empirical_variogram(
        std_v, value_col="z", directions=variogram_directions,
        n_pairs=variogram_n_pairs, n_bins=variogram_n_bins, seed=seed,
    )
    vgm = fit_anisotropic_variogram(
        emp, component_names=variogram_components, anisotropic=fit_anisotropy
    )

    # --- per-pixel σ raster over the full grid (same floor/ceiling as standardize) ---
    df_all = _attach_strip(build_predictor_frame(
        dh, slope, aspect, roughness, pc, stable_mask=None, add_pixel_index=True,
        extra_rasters=extra,
    ))
    sigma_map = evaluate_sigma_raster(model, df_all, shape)
    floor = float(std.attrs.get("sigma_floor", 0.0))
    ceiling = float(std.attrs.get("sigma_ceiling", np.inf))
    if floor > 0 or np.isfinite(ceiling):
        _finite = np.isfinite(sigma_map)
        diagnostics["n_pixels_sigma_ceiling_clipped"] = int(
            np.sum(_finite & (sigma_map > ceiling))
        )
        sigma_map = np.where(_finite, np.clip(sigma_map, floor, ceiling), sigma_map)
    # apply the same two-step standardization factor as `standardize`, so the
    # full-grid sigma raster is calibrated consistently with sigma_hat / z
    scale = float(std.attrs.get("sigma_scale", 1.0))
    if scale != 1.0:
        sigma_map = sigma_map * scale

    result: Dict[str, Any] = {
        "sigma_model": model,
        "sigma_raster": sigma_map,
        "empirical_variogram": emp,
        "variogram_model": vgm,
        "diagnostics": diagnostics,
        "standardized_frame": std,  # stable-terrain frame with sigma_hat + z
        "transform": transform,
        "crs": crs,
    }

    if foi_geometries:
        est = HeteroscedasticUncertaintyEstimator(sigma_map, transform, vgm)
        foi_unc = {}
        for name, geom in foi_geometries.items():
            if areal_method == "mc":
                foi_unc[name] = est.areal_uncertainty_mc(
                    geom, n_realizations=mc_realizations, seed=seed
                )
            else:
                foi_unc[name] = est.areal_uncertainty_pairwise(
                    geom, n_pairs=n_pairs, seed=seed
                )
        result["foi_uncertainty"] = foi_unc

    return result
