"""Regression tests for the alignment RMSE/fitness speedups (2026-07-16).

`_compute_rmse_fitness` was the bottleneck in `align_point_clouds`: a
single-threaded `cKDTree.query` over millions of source points against the full
target ran for minutes. The fix subsamples the *source* (query) cloud, runs the
query with `workers=-1`, and keeps the target at full density. These tests pin
that behaviour:

  * subsampling the source is unbiased for RMSE and fitness;
  * `num_inliers` is rescaled to the FULL source count (not the subsample);
  * the source subsample is seeded -> deterministic, and the pre/post calls use
    the same points so the auto-revert comparison stays apples-to-apples;
  * empty clouds are guarded;
  * the target is left at full density by default (subsampling it biases
    nearest-neighbour distances upward, inflating RMSE);
  * the macOS OpenMP import-order guard precedes the PDAL-importing submodule.
"""
import importlib.util
import inspect
import pathlib

import numpy as np
import pytest


def _fn():
    """Import the function under test, skipping if optional deps are missing."""
    align = pytest.importorskip("topochange.alignment")
    return align._compute_rmse_fitness


def _make_pair(n_tgt=200_000, n_src=120_000, z_offset=0.20, seed=42):
    """A source cloud that samples the same sloped, noisy surface as the target,
    displaced by a known vertical offset (removable by a corrective transform)."""
    rng = np.random.default_rng(seed)
    xy_t = rng.uniform(0, 100, size=(n_tgt, 2))
    z_t = 0.05 * xy_t[:, 0] + rng.normal(0, 0.03, n_tgt)
    tgt = np.column_stack([xy_t, z_t])
    xy_s = rng.uniform(0, 100, size=(n_src, 2))
    z_s = 0.05 * xy_s[:, 0] + rng.normal(0, 0.03, n_src) + z_offset
    src = np.column_stack([xy_s, z_s])
    return src, tgt


def test_source_subsampling_is_unbiased_and_rescales_inliers():
    f = _fn()
    src, tgt = _make_pair()
    ident = np.eye(4)
    rmse_full, fit_full, ninl_full = f(src, tgt, ident, 1.0, max_source=None)
    rmse_cap, fit_cap, ninl_cap = f(src, tgt, ident, 1.0, max_source=50_000, seed=0)
    # RMSE and fitness are statistics: a 50k subsample tracks the full cloud.
    assert abs(rmse_cap - rmse_full) / rmse_full < 0.05
    assert abs(fit_cap - fit_full) / fit_full < 0.05
    # num_inliers is rescaled to the FULL source count, not the 50k subsample.
    assert ninl_cap > 50_000
    assert abs(ninl_cap - fit_cap * len(src)) <= 1
    assert 0.9 * ninl_full < ninl_cap < 1.1 * ninl_full


def test_seed_makes_pre_post_comparable_and_reproducible():
    f = _fn()
    src, tgt = _make_pair()
    ident = np.eye(4)
    a = f(src, tgt, ident, 1.0, max_source=40_000, seed=5)
    b = f(src, tgt, ident, 1.0, max_source=40_000, seed=5)
    assert a == b                                   # deterministic for a fixed seed
    # The corrective transform lowers RMSE on the SAME seeded subset, so the
    # auto-revert gate (post_rmse >= pre_rmse) keeps the alignment.
    corrective = np.eye(4)
    corrective[2, 3] = -0.20
    pre = f(src, tgt, ident, 1.0, max_source=40_000, seed=0)
    post = f(src, tgt, corrective, 1.0, max_source=40_000, seed=0)
    assert post[0] < pre[0]


def test_empty_clouds_guarded():
    f = _fn()
    src, tgt = _make_pair(n_tgt=1000, n_src=1000)
    assert f(np.empty((0, 3)), tgt, np.eye(4), 1.0) == (float("inf"), 0.0, 0)
    assert f(src, np.empty((0, 3)), np.eye(4), 1.0) == (float("inf"), 0.0, 0)


def test_target_kept_full_density_by_default():
    """Subsampling the target inflates nearest-neighbour distances (and RMSE);
    this documents why max_target defaults to None."""
    f = _fn()
    src, tgt = _make_pair()
    corrective = np.eye(4)
    corrective[2, 3] = -0.20
    full = f(src, tgt, corrective, 1.0, max_source=40_000, seed=0)[0]
    thinned = f(src, tgt, corrective, 1.0, max_source=40_000, max_target=20_000, seed=0)[0]
    assert thinned > full


def test_signature_defaults():
    f = _fn()
    sig = inspect.signature(f)
    assert sig.parameters["max_source"].default == 500_000
    assert sig.parameters["max_target"].default is None
    assert sig.parameters["seed"].default == 0


def test_init_openmp_guard_precedes_pointcloud_import():
    """The KMP escape hatch and the early small_gicp import must both appear
    before the PDAL-importing `.pointcloud` submodule import, or the macOS
    dual-libomp abort returns. Source-level check (no heavy deps needed)."""
    spec = importlib.util.find_spec("topochange")
    if spec is None or not spec.origin:
        pytest.skip("topochange not importable")
    text = pathlib.Path(spec.origin).read_text()
    i_kmp = text.find("KMP_DUPLICATE_LIB_OK")
    i_sgicp = text.find("import small_gicp")
    i_pc = text.find("from .pointcloud")
    assert i_pc != -1
    assert 0 <= i_kmp < i_pc, "KMP_DUPLICATE_LIB_OK must be set before .pointcloud"
    assert 0 <= i_sgicp < i_pc, "small_gicp must import before PDAL (.pointcloud)"


def test_num_threads_defaults_to_one_on_macos(monkeypatch):
    """macOS defaults to a single small_gicp thread (dual-libomp livelock);
    other platforms keep the multi-core default; explicit values still win."""
    align = pytest.importorskip("topochange.alignment")
    monkeypatch.setattr(align.sys, "platform", "darwin")
    assert align._default_num_threads() == 1
    assert align.RegistrationConfig().num_threads == 1
    monkeypatch.setattr(align.sys, "platform", "linux")
    assert align._default_num_threads() >= 1
    # An explicit num_threads is never overridden by the platform default.
    assert align.RegistrationConfig(num_threads=6).num_threads == 6
