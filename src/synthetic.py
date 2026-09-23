"""Controlled synthetic skewness sweep helpers (Section S1).

Order-preserving remapping of a dataset's ratings to a target skewness, with the
mean softly anchored and the variance left free (variance compression is the
mechanism under test). Groups are defined by quantiles so |H|=|G| stays fixed
across skew levels, removing the group-size confound.
"""
import numpy as np
from scipy.stats import skew as scipy_skew

W_MEAN, W_SKEW, W_ENT = 5.0, 1.0, 1e-3


def _moments(p, v):
    m = np.sum(p * v)
    var = np.sum(p * (v - m) ** 2)
    sk = np.sum(p * (v - m) ** 3) / (var ** 1.5 + 1e-12)
    return m, var, sk


def _target_proportions(values, p_base, m0, target_skew):
    from scipy.optimize import minimize
    v = values.astype(float)

    def obj(p):
        p = np.clip(p, 1e-9, None); p = p / p.sum()
        m, var, sk = _moments(p, v)
        ent = -np.sum(p * np.log(p + 1e-12))
        return (W_SKEW * (sk - target_skew) ** 2 + W_MEAN * (m - m0) ** 2 - W_ENT * ent)

    cons = [{"type": "eq", "fun": lambda p: np.sum(p) - 1.0}]
    bnds = [(1e-6, 1.0)] * len(v)
    best = None
    for trial in range(5):
        x0 = (np.asarray(p_base, float) if trial == 0
              else np.random.default_rng(trial).dirichlet(np.ones(len(v))))
        try:
            r = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                         options={"maxiter": 1000, "ftol": 1e-12})
            p = np.clip(r.x, 1e-9, None); p /= p.sum()
            val = obj(p)
            if best is None or val < best[0]:
                best = (val, p)
        except Exception:
            pass
    return best[1] if best is not None else np.asarray(p_base, float)


def _quantile_map(train_df, test_df, values, p_target, seed):
    rng = np.random.default_rng(seed)
    tr, te = train_df.copy(), test_df.copy()
    nt = len(tr)
    allr = np.concatenate([tr.rating.to_numpy(float), te.rating.to_numpy(float)])
    jitter = allr + rng.uniform(-0.24, 0.24, size=len(allr))
    order = np.argsort(jitter, kind="mergesort")
    pct = np.empty(len(allr)); pct[order] = (np.arange(len(allr)) + 0.5) / len(allr)
    cum = np.cumsum(p_target)
    idx = np.clip(np.searchsorted(cum, pct, side="left"), 0, len(values) - 1)
    mapped = values[idx]
    tr["rating"] = mapped[:nt]; te["rating"] = mapped[nt:]
    return tr, te


def reshape_to_target_skew(train_df, test_df, target_skew, seed=0):
    """Return (train, test, achieved) with ratings remapped to target_skew."""
    ratings = train_df["rating"].to_numpy(float)
    values = np.unique(ratings)
    p_base = np.array([np.mean(ratings == v) for v in values], float)
    m0 = float(np.mean(ratings))
    p_target = _target_proportions(values, p_base, m0, target_skew)
    tr, te = _quantile_map(train_df, test_df, values, p_target, seed=seed)
    pooled = np.concatenate([tr.rating.to_numpy(float), te.rating.to_numpy(float)])
    achieved = {"skew": float(scipy_skew(pooled)), "var": float(np.var(pooled))}
    return tr, te, achieved


def quantile_groups(df, fraction=0.10):
    """Harsh = bottom `fraction` of users by positivity rho_u, generous = top."""
    item_avg = df.groupby("item_id")["rating"].mean()
    d = df.copy()
    d["above"] = d.rating > d.item_id.map(item_avg)
    rho = d.groupby("user_id")["above"].mean()
    lo, hi = rho.quantile(fraction), rho.quantile(1 - fraction)
    g = {}
    for u, r in rho.items():
        g[u] = "harsh" if r <= lo else "generous" if r >= hi else "neutral"
    return g
