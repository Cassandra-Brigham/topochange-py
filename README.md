# topochange

**Geostatistical uncertainty estimation for airborne lidar topographic differencing**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

`topochange` quantifies spatially correlated uncertainty in lidar topographic change detection. It decomposes vertical differencing error into bias, correlated, and uncorrelated components with nested variogram models, then propagates that error structure over user-defined regions of interest. The outputs are a standard deviation for the mean change in any polygon, a per-pixel (heteroscedastic) σ map, or a volume change with its uncertainty.

The package runs on the [OpenTopography](https://opentopography.org) platform and standalone, through the Jupyter notebooks in this repository.

> Manuscript:
> Brigham, C., Scott, C., Arrowsmith, R., Phan, M., DeWitt, J., Palaseanu-Lovejoy, M., Nandigam, V., Stoker, J., Anderson, S. W., Gesch, D. B., Crosby, C. J., & Beckley, M. (2026). Geostatistical error analysis in airborne lidar topographic differencing: Workflow for multi-scale uncertainty estimation of common error sources. *Earth and Space Science*.

## Scientific background

Vertical topographic differencing (pixel-by-pixel subtraction of DEMs collected at different times) underpins studies of landscape change from wildfires, landslides, erosion, tectonics, and vegetation dynamics. The measured change combines true surface change with error introduced during acquisition, processing, and alignment. That error is spatially structured at several scales:

- Short range (meters to tens of meters): sensor noise, point misclassification, geometric distortion on steep slopes
- Mid range (hundreds of meters): flight-line banding, topographically correlated georeferencing error
- Long range (kilometers): calibration biases, vertical datum and geoid mismatches

`topochange` treats the net differencing error as a spatially correlated random field (Matheron, 1965) and fits nested variogram models to split the total variance into a nugget, one or more correlated components (each with a sill and range), and a systematic bias estimated over stable terrain. The fitted variogram is integrated over arbitrary polygons by Monte Carlo sampling of the covariance function (Rolstad et al., 2009; Hugonnet et al., 2022), which gives a standard deviation for the mean change that accounts for spatial correlation. For spatially varying error, the heteroscedastic module models σ as a function of terrain and point-cloud predictors (slope, roughness, canopy, scan geometry) and propagates it through a standardized correlogram. Variogram computation is Numba-accelerated: about 40× faster than scikit-gstat at 10,000 samples, with far lower memory use.

Key references: Matheron (1965), *Les variables régionalisées et leur estimation*; Rolstad et al. (2009), [doi:10.3189/002214309789470950](https://doi.org/10.3189/002214309789470950); Hugonnet et al. (2022), [doi:10.1109/JSTARS.2022.3188922](https://doi.org/10.1109/JSTARS.2022.3188922); Oliver & Webster (2015), *Basic Steps in Geostatistics*; Anderson (2019), [doi:10.1002/esp.4551](https://doi.org/10.1002/esp.4551).

## What's in the repo

```
topochange/
├── 1_DifferencingWorkflow_user_pointclouds.ipynb        # start from LAS/LAZ point clouds
├── 2a_DifferencingWorkflow_user_dems.ipynb              # start from GeoTIFF DEMs
├── 2b_DifferencingWorkflow_user_differencing_results.ipynb  # start from a difference GeoTIFF
├── 3_DifferencingWorkflow_download_data.ipynb           # download from OpenTopography, full pipeline
├── src/topochange/
│   ├── raster.py / rasterpair.py            # GeoTIFF I/O, CRS + datum reconciliation, differencing
│   ├── pointcloud.py / pointcloudpair.py    # LAS/LAZ via PDAL, registration, DEM generation
│   ├── alignment.py / alignment_utils.py    # ICP/GICP/VGICP registration (small_gicp)
│   ├── variogram.py / variogram_models.py   # empirical variograms, nested model fitting (AICc)
│   ├── composite_variogram.py               # composite/nested variogram models
│   ├── uncertainty.py                       # regional (areal) uncertainty propagation
│   ├── sigma_a_ci.py                        # confidence intervals for σ_A
│   ├── heteroscedastic.py                   # spatially varying σ: GAM model + anisotropic correlogram
│   ├── sigma_map.py                         # per-pixel σ maps: binned NMAD models, calibration, CV
│   ├── volume.py                            # volume change ± uncertainty over polygons
│   ├── stable_area_analysis.py              # interactive stable-area selection (ipyleaflet)
│   ├── data_access.py / pipeline_builder.py # OpenTopography API, PROJ pipelines
│   ├── crs_*.py, geoid_utils.py, unit_utils.py, time_utils.py, velocity_model_*.py
│   └── data/velocity_models_registry.yaml   # bundled crustal deformation model registry
├── tests/                                   # pytest suite (960 tests, 29 modules + 2 validation scripts)
├── environment.yml                          # complete conda environment (all extras)
├── pyproject.toml / requirements.txt
└── CITATION.cff
```

## Installation

Python ≥ 3.9. A conda environment is recommended; PDAL, GDAL, and PROJ install most reliably from conda-forge. The bundled `environment.yml` builds the whole environment (all extras plus test tooling) in one solve:

```bash
git clone https://github.com/Cassandra-Brigham/topochange-py.git
cd topochange-py
conda env create -f environment.yml
conda activate topochange
pip install -e . --no-deps   # --no-deps: everything is already in the env
```

Or assemble it manually and let pip resolve the pure-Python extras:

```bash
conda create -n topochange python=3.11
conda activate topochange
conda install -c conda-forge pdal python-pdal gdal rasterio pyproj geopandas rioxarray numba

pip install -e ".[all]"
```

Or pick individual extras instead of `[all]`:

```bash
pip install -e ".[interactive]"      # ipyleaflet/ipywidgets (notebook maps)
pip install -e ".[pointcloud]"       # pdal, laspy
pip install -e ".[heteroscedastic]"  # pygam (per-pixel σ modelling)
pip install -e ".[alignment]"        # small_gicp (ICP registration)
pip install -e ".[data_access]"      # boto3 (OpenTopography downloads)
pip install -e ".[converters]"       # xarray (velocity model conversion)
pip install -e ".[dev]"              # pytest, black, ruff
```

In Google Colab, the setup cell at the top of each notebook installs everything (condacolab, then PDAL, then topochange). The first run restarts the kernel once; re-run the cell and it picks up where it left off. Notebook 3 also needs an [OpenTopography API key](https://opentopography.org/).

## Quick start

### Difference two DEMs and propagate uncertainty over a polygon

```python
from topochange import (
    Raster, RasterPair, RasterDataHandler, GridVariogram,
    RegionalUncertaintyEstimator,
)

# 1. Load + difference (CRS/datum reconciliation handled by RasterPair)
pair = RasterPair(Raster.from_file("compare.tif"), Raster.from_file("reference.tif"))
results = pair.compute_difference(
    interpolation_method="tin", clip_to_overlap=True,
    output_path="difference.tif", verbose=True,
)

# 2. Fit nested variogram models to the bias-removed difference raster
rdh = RasterDataHandler("difference_bias_removed.tif", "m", 1.0)  # path, unit, resolution
rdh.load_raster()
gv = GridVariogram(rdh, n_realizations=1)
gv.run(
    area_side=250, samples_per_area=400, max_samples=10_000_000,
    bin_width=10, max_lag_multiplier=1/3, criterion="aicc", seed=42,
)

# 3. Regionalized uncertainty for a feature of interest (shapely polygon)
est = RegionalUncertaintyEstimator(
    raster_data_handler=rdh,
    variogram_analysis=gv,
    area_of_interest=polygon,
    fitted_model=gv.fitted_model,
)
est.calc_total_uncertainty(n_pairs=25_000, seed=42)
print(est.summary())
```

Stable areas for bias removal can be drawn interactively with `TopoMapInteractor` / `StableAreaRasterizer` / `StableAreaAnalyzer` (see notebooks 1, 2a, 2b).

### Volume change ± uncertainty

```python
from topochange import VolumeEstimator

vol = VolumeEstimator(est).compute()   # est from the previous step
print(vol.summary())                   # net/cut/fill volumes with σ_V = A · σ_ΔH
```

### Per-pixel (heteroscedastic) σ map

```python
from topochange import run_heteroscedastic_pipeline, write_sigma_geotiff

out = run_heteroscedastic_pipeline(
    dh_path="difference_bias_removed.tif",
    reference_dem_path="reference.tif",   # terrain predictors derived automatically
    las_path="compare.laz",               # optional: adds point-cloud predictors
)
write_sigma_geotiff(out["sigma_raster"], "sigma.tif", out["transform"], out["crs"])
```

The `sigma_map` module is the non-parametric alternative, using binned NMAD as in xDEM (Hugonnet et al.), with calibration diagnostics and spatial cross-validation. See `fit_binned_sigma_model`, `calibration_by_bin`, `cross_validate_sigma_models`, and `misregistration_diagnostic`.

## Notebook workflows

| Notebook | Start from | What it does |
|---|---|---|
| `1_…user_pointclouds` | LAS/LAZ point clouds | Metadata and CRS checks, ICP registration, DEM generation, differencing, variogram fit, uncertainty |
| `2a_…user_dems` | GeoTIFF DEMs | CRS/datum reconciliation, differencing, stable areas, variogram fit, uncertainty |
| `2b_…user_differencing_results` | A difference GeoTIFF | Skips differencing: stable-area selection, bias removal, variogram fit, per-feature uncertainty |
| `3_…download_data` | OpenTopography catalog | Query + download point clouds, then the full pipeline (API key required) |

Set the config cell at the top of each notebook (`DATA_PATH = "your/path/here"`, file names, CRS/epoch/geoid metadata) and run top to bottom.

## API overview

| Task | Entry points |
|---|---|
| Rasters & differencing | `Raster`, `RasterPair` |
| Point clouds | `PointCloud`, `PointCloudPair` |
| Registration | `LandscapeAligner`, `RegistrationConfig` (incl. `sample_method="head"\|"random"` subsampling), `align_point_clouds` |
| Variograms | `RasterDataHandler`, `SingleVariogram`, `GridVariogram`, `CompositeVariogramModel`, `MODEL_REGISTRY` |
| Areal uncertainty | `RegionalUncertaintyEstimator`, `DerivativeUncertaintyEstimator`, `sigma_a_ci` |
| Per-pixel σ | `heteroscedastic` (GAM path), `sigma_map` (binned NMAD path), `write_sigma_geotiff` |
| Volume | `VolumeEstimator`, `polygon_volume` |
| Stable areas | `TopoMapInteractor`, `StableAreaRasterizer`, `StableAreaAnalyzer` |
| CRS / datums | `CRSHistory`, `CRSState`, `build_vertical_pipeline`, `geoid_utils` |
| Data access | `DataAccess`, `OpenTopographyQuery`, `GetDEMs` |

Backward-compatible aliases: `VariogramAnalysis`, `FittedVariogramModel`, `EmpiricalVariogram`, `StatisticalAnalysis`.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

Tests requiring optional dependencies (PDAL, small_gicp, GDAL) skip automatically when those aren't installed; ~860 of the 960 tests run without them. See `tests/README.md` for the layout.

## Citation

```bibtex
@software{brigham2026topochange,
  author       = {Brigham, Cassandra and Scott, Chelsea and Arrowsmith, Ramon and
                  Phan, Minh and DeWitt, Jessica and Palaseanu-Lovejoy, Monica and
                  Nandigam, Viswanath and Stoker, Jason and Anderson, Scott Wallace and
                  Gesch, Dean B and Crosby, Christopher J and Beckley, Matthew},
  title        = {topochange},
  version      = {1.0.0},
  year         = {2026},
  url          = {https://github.com/Cassandra-Brigham/topochange-py},
  license      = {MIT}
}
```

And the accompanying manuscript:

> Brigham, C., Scott, C., Arrowsmith, R., Phan, M., DeWitt, J., Palaseanu-Lovejoy, M., Nandigam, V., Stoker, J., Anderson, S. W., Gesch, D. B., Crosby, C. J., & Beckley, M. (2026). Geostatistical error analysis in airborne lidar topographic differencing: Workflow for multi-scale uncertainty estimation of common error sources. *Earth and Space Science*.

## Funding and acknowledgements

This work was supported by the U.S. Geological Survey Powell Center (Grant G23AC00336) and the National Science Foundation (Grants 2410800, 2410799, 2410801). We thank the members of the USGS Powell Center working group on Topographic Change: Pete Chirico, Kara Doran, Zhong Lu, Carrie Middleton, Aldo Plascencia, Giulia Sofia, and Joe Wheaton. The on-demand implementation runs on [OpenTopography](https://opentopography.org), with compute resources from the San Diego Supercomputer Center.

## License

[MIT](https://opensource.org/licenses/MIT)
