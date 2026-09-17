"""Quick P0-2 / P0-3 checks on grf_dataset fields.

Lightweight: needs only numpy / scipy / imageio (no rasterio / numba), so it
runs in a notebook fast.  It MIRRORS the math of the shipped changes:
  * P0-2 detrending -> SingleVariogram._fit_polynomial_trend (verbatim)
  * P0-3 bias SE    -> RegionalUncertaintyEstimator.estimate_std_mean_monte_carlo
  * Matheron binning + Cressie(1985)-weighted curve_fit (spherical proxy fit)

The detrend (OFF/ON) and bias-SE (old/new) numbers are RELATIVE comparisons and
are meaningful for any family; the analytic "true" sigma_A reference is exact
only for Spherical fields.  For rigorous, family-aware coverage use
``validate_p1_coverage.py`` (which drives the real package).

Notebook use
------------
    import sys; sys.path.insert(0, "tests")
    import validate_p0_grf as v0
    files = v0.list_datasets("Spherical_r100_s*_n0_*")      # choose datasets
    df = v0.run(files, add_ramp=True, detrend_order=1)       # returns a DataFrame
"""
import os, re, glob, math
import numpy as np
import imageio.v3 as iio
from scipy.optimize import curve_fit

DATA = os.environ.get(
    "TOPO_GRF_DIR",
    os.path.join(os.path.dirname(__file__), "..", "synthetic_benchmark", "grf_dataset"),
)
FEATURE = (400, 400, 600, 600)     # centre square (feature of interest)
DOMAIN = (0, 0, 1000, 1000)        # stable-area proxy

def list_datasets(pattern="*", data=None):
    """Return sorted absolute paths of grf tifs matching a glob pattern.

    Examples: list_datasets("Spherical_r100_*"), list_datasets("*planar-sl0-01*").
    """
    data = data or DATA
    if not pattern.endswith(".tif"):
        pattern = pattern + ".tif"
    return sorted(glob.glob(os.path.join(data, pattern)))

# --- shipped detrend math (identical to SingleVariogram._fit_polynomial_trend) ---
def _design(coords, order, center, scale):
    x = (coords[:, 0] - center[0]) / scale[0]; y = (coords[:, 1] - center[1]) / scale[1]
    cols = [np.ones_like(x)]
    if order >= 1: cols += [x, y]
    if order >= 2: cols += [x * x, x * y, y * y]
    return np.column_stack(cols)

def fit_polynomial_trend(coords, values, order):
    center = coords.mean(0); scale = coords.std(0); scale[scale == 0] = 1.0
    X = _design(coords, order, center, scale)
    beta, *_ = np.linalg.lstsq(X, values, rcond=None)
    resid = values - X @ beta
    tv = float(np.var(values))
    return resid, (0.0 if tv <= 0 else 1.0 - float(np.var(resid)) / tv)

# --- empirical variogram + spherical proxy fit + sigma_A integral ---
def _empirical(coords, values, bin_width=15.0, max_lag=350.0, min_pairs=10):
    iu = np.triu_indices(len(coords), 1)
    dd = np.linalg.norm(coords[iu[0]] - coords[iu[1]], axis=1)
    d2 = (values[iu[0]] - values[iu[1]]) ** 2
    k = dd <= max_lag; dd, d2 = dd[k], d2[k]; idx = (dd / bin_width).astype(int)
    L, G, C = [], [], []
    for b in range(int(np.ceil(max_lag / bin_width)) + 1):
        m = idx == b; n = int(m.sum())
        if n >= min_pairs:
            L.append(dd[m].mean()); G.append(d2[m].sum() / (2 * n)); C.append(n)
    return np.array(L), np.array(G), np.array(C, float)

def _sph(h, a): x = np.minimum(np.asarray(h, float) / a, 1.0); return 1.5 * x - 0.5 * x ** 3
def _gamma(h, c0, s, a): return c0 + s * _sph(h, a)

def _fit(L, G, C):
    sigma = 1.0 / np.sqrt(np.maximum(C / np.maximum(G, 1e-12) ** 2, 1e-12))  # Cressie 1985
    smax = float(G.max()); thr = G >= 0.63 * smax
    a0 = float(L[np.argmax(thr)]) if thr.any() else L[-1] / 2
    try:
        p, _ = curve_fit(_gamma, L, G, p0=[0.05 * smax, smax, max(a0, L[1])], sigma=sigma,
                         bounds=([0, 1e-6, L[1] * 0.1], [smax + 1e-9, 5 * smax, L[-1] * 3]), maxfev=20000)
    except Exception:
        p = [0.05 * smax, smax, a0]
    return p  # c0, s, a

def _sigma_A(box, c0, s, a, n=40000, seed=0):
    r = np.random.default_rng(seed); x0, y0, x1, y1 = box
    X = np.c_[r.uniform(x0, x1, n), r.uniform(y0, y1, n)]
    Y = np.c_[r.uniform(x0, x1, n), r.uniform(y0, y1, n)]
    h = np.linalg.norm(X - Y, axis=1)
    return math.sqrt(max(float(np.mean((c0 + s) - _gamma(h, c0, s, a))), 0.0))

def _sample(field, n, seed):
    ny, nx = field.shape; r = np.random.default_rng(seed)
    ij = r.choice(ny * nx, min(n, ny * nx), replace=False); yy, xx = np.divmod(ij, nx)
    return np.c_[xx.astype(float), yy.astype(float)], field[yy, xx].astype(float)

def _parse(fname):
    m1, m2 = re.search(r"_r(\d+)_", fname), re.search(r"_s([0-9\-]+)_", fname)
    fam = fname.split("_")[0]
    if not (m1 and m2):
        return fam, None, None
    s = m2.group(1).replace("-", ".")
    if s.startswith("."): s = "0" + s
    return fam, int(m1.group(1)), float(s)

def check_field(path, add_ramp=True, detrend_order=1, ramp_slope=(0.002, 0.001),
                n_sample=800, seed=1, feature=FEATURE, domain=DOMAIN):
    """Return P0-2 (detrend off/on) and P0-3 (bias SE old/new) metrics for one field."""
    fam, rng_, sill = _parse(os.path.basename(path))
    field = np.asarray(iio.imread(path), float)
    coords, vals0 = _sample(field, n_sample, seed)
    ramp = (lambda c: ramp_slope[0] * (c[:, 0] - 500) + ramp_slope[1] * (c[:, 1] - 500)) if add_ramp else (lambda c: 0.0)

    # P0-2: sigma_A over feature, detrend OFF vs ON
    def _sa(order):
        v = vals0 + ramp(coords)
        ve = None
        if order > 0:
            v, ve = fit_polynomial_trend(coords, v, order)
        c0, s, a = _fit(*_empirical(coords, v))
        return _sigma_A(feature, c0, s, a, seed=seed), (a, s, ve)
    off, (a_off, s_off, _) = _sa(0)
    on, (a_on, s_on, ve) = _sa(detrend_order)

    # P0-3: bias SE over stable/domain area -- old iid vs new correlation-aware
    c0, s, a = _fit(*_empirical(coords, vals0))
    new_bias = _sigma_A(domain, c0, s, a, seed=seed + 3)
    old_bias = float(np.std(field)) / math.sqrt(field.size)
    true_feat = _sigma_A(feature, 0.0, sill, rng_, seed=9) if (fam == "Spherical" and sill) else None

    return dict(file=os.path.basename(path), family=fam, true_range=rng_, true_sill=sill,
                sigmaA_detrend_off=off, sigmaA_detrend_on=on,
                detrend_ratio=off / max(on, 1e-12), trend_var_explained=ve,
                sigmaA_feature_true=true_feat,
                bias_se_old_iid=old_bias, bias_se_new=new_bias,
                bias_new_over_old=new_bias / max(old_bias, 1e-12))

def run(files, add_ramp=True, detrend_order=1, **kw):
    """Run check_field over a chosen list of files; print a table and return a DataFrame."""
    rows = [check_field(f, add_ramp=add_ramp, detrend_order=detrend_order, **kw) for f in files]
    for r in rows:
        tf = f"{r['sigmaA_feature_true']:.3f}" if r["sigmaA_feature_true"] is not None else "  -  "
        print(f"{r['file'][:32]:32s} true={tf}  detrend off={r['sigmaA_detrend_off']:.3f} "
              f"on={r['sigmaA_detrend_on']:.3f} (x{r['detrend_ratio']:.2f})  "
              f"biasSE old={r['bias_se_old_iid']:.5f} new={r['bias_se_new']:.4f} (x{r['bias_new_over_old']:.1f})")
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except Exception:
        return rows

if __name__ == "__main__":
    sel = list_datasets("Spherical_r100_s*_n0_*")[:4]
    run(sel, add_ramp=True, detrend_order=1)
