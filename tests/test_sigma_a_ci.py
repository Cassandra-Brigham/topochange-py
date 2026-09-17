"""Tests for topochange.sigma_a_ci (sigma_A confidence-interval constructions).

Validates the self-contained numerical pieces:
  * circulant simulation reproduces the fitted covariance,
  * the bias-correcting log-basic interval is calibrated,
  * the Buckland model-averaged SE formula,
  * the Gaussian experimental-variogram covariance kernel vs. Monte Carlo,
  * the Path-A bootstrap driver plumbing (ensemble draw, aggregation, seeding).

The full Path-A pipeline (make_estimate_fn -> GridVariogram -> Regional...) is a
thin wiring of already-tested classes and is exercised on real data, not here.
"""

import math

import numpy as np
import pytest

from topochange.composite_variogram import CompositeVariogramModel
from topochange import sigma_a_ci as ci


def _exp_nugget(sill, rng_, nugget):
    m = CompositeVariogramModel(["exponential"], include_nugget=True)
    m.set_params([sill, rng_, nugget])
    return m


# ----------------------------------------------------------------------
# 1. circulant simulation reproduces the fitted covariance/variogram
# ----------------------------------------------------------------------

def test_simulate_recovers_variogram():
    sill, rng_len, nugget = 0.8, 12.0, 0.2
    model = _exp_nugget(sill, rng_len, nugget)
    ny = nx = 96
    gen = np.random.default_rng(0)

    n_fields = 60
    lags = np.array([1, 2, 4, 8, 16])
    acc = np.zeros(len(lags))
    for _ in range(n_fields):
        f = ci.simulate_field_from_model(model, ny, nx, 1.0, gen)
        for i, d in enumerate(lags):
            diff = f[:, d:] - f[:, :-d]           # semivariance along x
            acc[i] += 0.5 * np.mean(diff ** 2)
    emp = acc / n_fields
    theo = model(lags.astype(float))              # includes nugget for h > 0

    # ensemble-mean empirical variogram should match the model within MC error
    rel = np.abs(emp - theo) / theo
    assert np.all(rel < 0.12), f"variogram mismatch: emp={emp}, theo={theo}, rel={rel}"

    # total sill (correlated + nugget) approached at long lag (exponential is
    # asymptotic, so use a distance many range-lengths out)
    long_lag = model(np.array([400.0]))[0]
    assert abs(long_lag - (sill + nugget)) < 1e-6


def test_simulate_rejects_unbounded():
    m = CompositeVariogramModel(["power"], include_nugget=True)
    m.set_params([0.01, 1.5, 0.1])
    with pytest.raises(ValueError):
        ci.simulate_field_from_model(m, 16, 16, 1.0, np.random.default_rng(0))


# ----------------------------------------------------------------------
# 2. the bias-correcting log interval is calibrated
# ----------------------------------------------------------------------

def test_log_basic_interval_coverage():
    rng = np.random.default_rng(1)
    tau, beta, s = 1.0, math.log(1.06), 0.15
    B, T = 199, 6000
    cov68 = cov95 = 0
    for _ in range(T):
        hat = tau * math.exp(rng.normal(beta, s))
        star = hat * np.exp(rng.normal(beta, s, B))
        (lo, hi), _ = ci.log_basic_interval(hat, star, 0.32)
        (lo9, hi9), _ = ci.log_basic_interval(hat, star, 0.05)
        cov68 += lo <= tau <= hi
        cov95 += lo9 <= tau <= hi9
    c68, c95 = 100 * cov68 / T, 100 * cov95 / T
    assert 64 <= c68 <= 72, f"68% coverage {c68:.1f}"
    assert 92 <= c95 <= 98, f"95% coverage {c95:.1f}"


# ----------------------------------------------------------------------
# 3. Buckland model-averaged SE
# ----------------------------------------------------------------------

def test_buckland_two_model():
    s_bar, se = ci.buckland_model_averaged_se(
        point_estimates=[1.0, 1.2], within_sd=[0.1, 0.15], weights=[0.7, 0.3]
    )
    assert abs(s_bar - 1.06) < 1e-12
    # 0.7*sqrt(0.01+0.06^2) + 0.3*sqrt(0.0225+0.14^2)
    expect = 0.7 * math.sqrt(0.01 + 0.06 ** 2) + 0.3 * math.sqrt(0.0225 + 0.14 ** 2)
    assert abs(se - expect) < 1e-12
    # between-model dispersion must widen SE beyond the weighted within-SD
    assert se > 0.7 * 0.1 + 0.3 * 0.15


# ----------------------------------------------------------------------
# 4. Gaussian experimental-variogram covariance kernel vs Monte Carlo
# ----------------------------------------------------------------------

def test_experimental_variogram_covariance_matches_mc():
    rng = np.random.default_rng(3)
    n = 20
    coords = rng.uniform(0, 10, size=(n, 2))
    a = 3.0                                   # exponential, unit sill, no nugget
    gamma = lambda h: 1.0 - np.exp(-np.asarray(h, float) / a)
    cov_fn = lambda h: np.exp(-np.asarray(h, float) / a)

    bin_edges = np.array([0.0, 2.0, 4.0, 6.0, 9.0])
    Sigma, _, _ = ci.experimental_variogram_covariance_gaussian(coords, gamma, bin_edges)

    # Monte-Carlo: simulate Gaussian fields on these points, bin the Matheron
    # variogram, take the empirical covariance of the bin estimates.
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    C = cov_fn(D)
    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
    iu, ju = np.triu_indices(n, k=1)
    dvec = D[iu, ju]
    which = np.digitize(dvec, bin_edges) - 1
    nb = len(bin_edges) - 1

    M = 30000
    ghat = np.empty((M, nb))
    for m in range(M):
        z = L @ rng.standard_normal(n)
        sq = 0.5 * (z[iu] - z[ju]) ** 2
        for k in range(nb):
            sel = which == k
            ghat[m, k] = sq[sel].mean() if sel.any() else np.nan
    Sig_mc = np.cov(ghat, rowvar=False)

    # compare well-populated entries (diagonal + neighbours)
    for k in range(nb):
        assert abs(Sigma[k, k] - Sig_mc[k, k]) / Sig_mc[k, k] < 0.12, (
            f"diag {k}: closed={Sigma[k,k]:.4e} mc={Sig_mc[k,k]:.4e}")
    for k in range(nb - 1):
        cl, mc = Sigma[k, k + 1], Sig_mc[k, k + 1]
        assert abs(cl - mc) / abs(mc) < 0.20, (
            f"offdiag {k},{k+1}: closed={cl:.4e} mc={mc:.4e}")


# ----------------------------------------------------------------------
# 5. Path-A bootstrap driver plumbing (no heavy pipeline)
# ----------------------------------------------------------------------

def _template(ny=32, nx=32):
    return {"ny": ny, "nx": nx, "res": 1.0, "transform": None,
            "crs": None, "valid": np.ones((ny, nx), bool)}


def test_field_bootstrap_driver_and_seeding():
    model = _exp_nugget(0.7, 10.0, 0.3)
    tmpl = _template()

    def fake_estimate(field):
        v = field[field != ci.NODATA]
        return {"sigma_a": float(np.std(v)), "sigma_tot": float(np.std(v)) * 1.1}

    r1 = ci.field_bootstrap_sigma_a(model, tmpl, fake_estimate, sigma_a_hat=0.9,
                                    sigma_tot_hat=1.0, B=24, seed=42)
    r2 = ci.field_bootstrap_sigma_a(model, tmpl, fake_estimate, sigma_a_hat=0.9,
                                    sigma_tot_hat=1.0, B=24, seed=42)

    assert r1["samples"].size == 24
    assert set(["ci68", "ci95", "ci68_percentile", "sigma_tot"]).issubset(r1)
    assert r1["ci68"][0] < r1["ci68"][1]
    # reproducible across runs with the same seed
    np.testing.assert_allclose(r1["samples"], r2["samples"])
    # sigma_tot block present and ordered
    assert r1["sigma_tot"]["ci95"][0] < r1["sigma_tot"]["ci95"][1]


def test_field_bootstrap_ensemble_source():
    m1 = _exp_nugget(0.8, 10.0, 0.2)
    m2 = _exp_nugget(0.6, 14.0, 0.1)
    ensemble = [(m1, 0.6, np.array([[0.8, 10.0, 0.2]])),
                (m2, 0.4, np.array([[0.6, 14.0, 0.1]]))]
    tmpl = _template(64, 64)
    fake = lambda f: {"sigma_a": float(np.std(f[f != ci.NODATA])), "sigma_tot": 1.0}
    r = ci.field_bootstrap_sigma_a(ensemble, tmpl, fake, sigma_a_hat=0.9, B=20, seed=1)
    assert r["samples"].size == 20
    assert np.all(np.isfinite(r["samples"]))
