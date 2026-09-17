"""topographic change detection and uncertainty quantification."""

__version__ = "1.0.0"

# ── OpenMP import-order guard (macOS) ────────────────────────────────────────
# small_gicp and PDAL each bundle their own OpenMP runtime (libomp). On macOS,
# whichever runtime initializes second calls abort() ("OMP: Error #15:
# Initializing libomp, but found libomp already initialized"), killing the
# process outright. This manifested as the point-cloud notebooks crashing the
# Jupyter kernel the moment alignment ran, because ``.pointcloud`` imports PDAL
# below *before* ``.alignment`` imports small_gicp. Two defenses, applied before
# any submodule import so the ordering is fixed at package-import time:
#   1. KMP_DUPLICATE_LIB_OK=TRUE, Intel/LLVM's documented escape hatch that
#      lets the duplicate runtime load instead of aborting (set only if unset,
#      so an explicit user/environment setting is never overridden).
#   2. Import small_gicp first, so *its* libomp is initialized before PDAL's
#      (the ordering small_gicp is validated against upstream).
# Both are scoped to darwin: on Linux the GNU OpenMP runtime coexists fine and
# ignores KMP_DUPLICATE_LIB_OK, so the early import would only add cost.
import os as _os
import sys as _sys
if _sys.platform == "darwin":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import small_gicp as _small_gicp  # noqa: F401  (early libomp load)
    except Exception:  # optional dependency; real handling lives in .alignment
        pass

from .raster import Raster
from .rasterpair import RasterPair

from .pointcloud import PointCloud
from .pointcloudpair import PointCloudPair

from .variogram import (
    RasterDataHandler,
    SingleVariogram,
    GridVariogram,
    KrigingLOOCVResult,
    AggregatedLOOCVResult,
    # backward-compatibility stubs
    VariogramAnalysis,
    FittedVariogramModel,
    EmpiricalVariogram,
    StatisticalAnalysis,
)
from .uncertainty import RegionalUncertaintyEstimator, DerivativeUncertaintyEstimator
from .volume import VolumeEstimator, VolumeResult, polygon_volume
from .variogram_models import MODEL_REGISTRY, VariogramModelRegistry
from .composite_variogram import CompositeVariogramModel

from .heteroscedastic import (
    ClassificationSpec,
    HeteroscedasticSigmaModel,
    AnisotropicCompositeVariogram,
    HeteroscedasticUncertaintyEstimator,
    fit_sigma_model,
    standardize,
    extract_pointcloud_predictors,
    directional_empirical_variogram,
    fit_anisotropic_variogram,
    run_heteroscedastic_pipeline,
)

from .sigma_map import (
    BinnedSigmaModel,
    CallableSigmaModel,
    BinnedBiasModel,
    fit_binned_sigma_model,
    fit_binned_bias_model,
    misregistration_diagnostic,
    nd_binning,
    mixture_nmad,
    write_sigma_geotiff,
    calibration_by_bin,
    spatial_block_folds,
    cross_validate_sigma_models,
    patch_validation,
    uniform_edges,
    trimmed_nmad_scale,
)

from .stable_area_analysis import (
    TopoMapInteractor,
    StableAreaRasterizer,
    StableAreaAnalyzer,
)

from .crs_history import CRSHistory
from .pipeline_builder import CRSState, build_vertical_pipeline

# data_access needs GDAL (osgeo) and boto3, which are optional extras. When
# they are absent the three names below are simply not exported, so they are
# appended to __all__ only on success -- otherwise `from topochange import *`
# would raise AttributeError on a core-only install.
_DATA_ACCESS_EXPORTS: list[str] = []
try:
    from .data_access import DataAccess, OpenTopographyQuery, GetDEMs
except ImportError:
    pass
else:
    _DATA_ACCESS_EXPORTS = ["DataAccess", "OpenTopographyQuery", "GetDEMs"]

from .alignment import (
    LandscapeAligner,
    RegistrationConfig,
    RegistrationResult,
    RegistrationMethod,
    align_point_clouds,
)

from .alignment_utils import (
    load_points_from_las,
    save_transformed_las,
    compute_alignment_quality,
    PointCloudPreprocessor,
    AlignmentQualityMetrics,
)

__all__ = [
    "__version__",
    "Raster",
    "RasterPair",
    "PointCloud",
    "PointCloudPair",
    "LandscapeAligner",
    "RegistrationConfig",
    "RegistrationResult",
    "RegistrationMethod",
    "align_point_clouds",
    "PointCloudPreprocessor",
    "AlignmentQualityMetrics",
    "load_points_from_las",
    "save_transformed_las",
    "compute_alignment_quality",
    "RasterDataHandler",
    "SingleVariogram",
    "GridVariogram",
    "KrigingLOOCVResult",
    "AggregatedLOOCVResult",
    "MODEL_REGISTRY",
    "VariogramModelRegistry",
    "CompositeVariogramModel",
    "RegionalUncertaintyEstimator",
    "DerivativeUncertaintyEstimator",
    "VolumeEstimator",
    "VolumeResult",
    "polygon_volume",
    "ClassificationSpec",
    "BinnedSigmaModel",
    "BinnedBiasModel",
    "fit_binned_sigma_model",
    "fit_binned_bias_model",
    "misregistration_diagnostic",
    "nd_binning",
    "mixture_nmad",
    "write_sigma_geotiff",
    "calibration_by_bin",
    "spatial_block_folds",
    "cross_validate_sigma_models",
    "patch_validation",
    "CallableSigmaModel",
    "uniform_edges",
    "trimmed_nmad_scale",
    "HeteroscedasticSigmaModel",
    "AnisotropicCompositeVariogram",
    "HeteroscedasticUncertaintyEstimator",
    "fit_sigma_model",
    "standardize",
    "extract_pointcloud_predictors",
    "directional_empirical_variogram",
    "fit_anisotropic_variogram",
    "run_heteroscedastic_pipeline",
    "TopoMapInteractor",
    "StableAreaRasterizer",
    "StableAreaAnalyzer",
    "CRSHistory",
    "CRSState",
    "build_vertical_pipeline",
    "VariogramAnalysis",
    "FittedVariogramModel",
    "EmpiricalVariogram",
    "StatisticalAnalysis",
]

__all__ += _DATA_ACCESS_EXPORTS

