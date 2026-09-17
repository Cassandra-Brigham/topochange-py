"""Package-level smoke tests.

Pins two things nothing else pins:
1. ``import topochange`` works and every name in ``__all__`` resolves
   (a stray top-level import of an optional dependency in the import chain
   would break this before any user code runs);
2. the bundled velocity-model registry YAML ships with the package and
   parses.
"""

import topochange

# data_access needs GDAL + boto3 + ipyleaflet (optional extras); __init__
# wraps its import in try/except, so these three names may legitimately be
# absent on a core-only install.
_OPTIONAL_EXPORTS = {"DataAccess", "OpenTopographyQuery", "GetDEMs"}


def test_version_string():
    assert isinstance(topochange.__version__, str)
    assert topochange.__version__


def test_all_exports_resolve():
    missing = [
        name for name in topochange.__all__
        if not hasattr(topochange, name) and name not in _OPTIONAL_EXPORTS
    ]
    assert missing == [], f"__all__ names that do not resolve: {missing}"


def test_new_modules_exported():
    # the reconciled API surface: volume + sigma_map + heteroscedastic
    for name in ("VolumeEstimator", "polygon_volume", "BinnedSigmaModel",
                 "fit_binned_sigma_model", "write_sigma_geotiff",
                 "HeteroscedasticSigmaModel", "run_heteroscedastic_pipeline"):
        assert hasattr(topochange, name), name


def test_bundled_registry_yaml_loads():
    from topochange.velocity_model_registry import load_registry

    models = load_registry()
    assert len(models) >= 10
    names = [m.name for m in models]
    assert len(names) == len(set(names)), "registry model names must be unique"
    assert all(isinstance(m.name, str) and m.name for m in models)
