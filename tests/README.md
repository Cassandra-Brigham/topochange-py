# topochange test suite

1001 tests across 30 `test_*.py` modules, plus 2 standalone benchmark scripts
(`validate_p0_grf.py`, `validate_p1_coverage.py` — `__main__`-driven, need
external GRF benchmark data, not collected by pytest).

## Running

```bash
pip install -e ".[dev]"
pytest            # from the repo root; pyproject.toml sets testpaths/pythonpath
pytest -q tests/test_volume.py            # one module
pytest --cov=topochange                   # with coverage
```

Tests requiring optional dependencies skip automatically (gates in
`skip_markers.py`): **pdal** (point-cloud I/O, DEM creation, integration
workflows), **small_gicp** (registration), **GDAL/osgeo** (download resume).
With none of those installed, 902 tests still run; nothing errors or fails
from missing dependencies or missing data — synthetic LAZ fixtures are
generated on the fly by `conftest.py` when pdal is available.

## Layout

| Area | Modules |
|---|---|
| Rasters & differencing | `test_raster_and_rasterpair`, `test_difference_edge_mask`, `test_compute_2d_difference_overwrite`, `test_dem_tin_continuous` |
| Point clouds & DEMs | `test_pointcloud_metadata`, `test_pointcloud_transformation`, `test_dem_creation`, `test_option1_integration` (pdal-gated) |
| Registration | `test_alignment`, `test_alignment_rmse`, `test_sample_method` |
| Variograms & fitting | `test_variogram_models`, `test_composite_variogram`, `test_variogram_analysis` |
| Areal uncertainty & CIs | `test_uncertainty`, `test_sigma_a_ci`, `test_ci_integration` |
| Per-pixel σ | `test_heteroscedastic`, `test_chm_extra_predictors`, `test_sigma_map` |
| Volume | `test_volume` |
| CRS / units / metadata | `test_crs_utils`, `test_metadata_propagation`, `test_audit_fixes` (unique M2/M8 regressions), `test_utils` |
| Data access & performance | `test_data_access_pipelines` (osgeo mocked), `test_performance_optimizations` |
| Review regressions | `test_review_2026_07_16` (also the only coverage of `stable_area_analysis`) |
| Package health | `test_package_smoke` (imports, `__all__`, bundled registry YAML) |

## Conventions

- No pytest markers are registered; select tests by path or `-k` expression.
- Shared skip gates and synthetic-data constants live in `skip_markers.py`
  and `conftest.py`.
- Statistical tests use fixed seeds; hand-computable expected values are
  preferred over tolerance-only assertions.
