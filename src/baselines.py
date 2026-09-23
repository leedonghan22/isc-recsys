"""Post-processing baselines for the S2 comparison.

Two monotonic calibrators (isotonic, piecewise-linear "linear spline") that
refine predicted ratings but preserve item order, and three re-rankers (MMR,
popularity penalty, FA*IR-style prefix quota) that change item order for
objectives other than user-side recall parity.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Monotonic calibrators (order-preserving -> ranking gap unchanged)
# --------------------------------------------------------------------------
def fit_isotonic(s, r):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=1.0, y_max=5.0, increasing=True,
                             out_of_bounds="clip")
    iso.fit(np.asarray(s, float).ravel(), np.asarray(r, float).ravel())
    return iso


def apply_isotonic(iso, s):
    return iso.predict(np.asarray(s, float).ravel())


def _monotone_knots(s, r, n_knots=30):
    s = np.asarray(s, float).ravel()
    r = np.asarray(r, float).ravel()
    x = np.unique(np.quantile(s, np.linspace(0, 1, max(4, n_knots))))
    if x.size < 4:
        x = np.linspace(s.min(), s.max(), 10)
    mids = (x[:-1] + x[1:]) / 2.0
    bin_id = np.digitize(s, mids)
    y = np.array([r[bin_id == k].mean() if np.any(bin_id == k) else np.nan
                  for k in range(x.size)])
    for k in range(1, y.size):
        if np.isnan(y[k]):
            y[k] = y[k - 1]
    for k in range(y.size - 2, -1, -1):
        if np.isnan(y[k]):
            y[k] = y[k + 1]
    y = np.clip(np.maximum.accumulate(y), 1.0, 5.0)
    return x, y


def fit_piecewise_linear(s, r, n_knots=30):
    return _monotone_knots(s, r, n_knots)


def apply_piecewise_linear(x, y, s):
    return np.clip(np.interp(np.asarray(s, float).ravel(), x, y,
                             left=y[0], right=y[-1]), 1.0, 5.0)


# --------------------------------------------------------------------------
# Re-rankers (operate on vanilla MF scores; change the top-K set)
# --------------------------------------------------------------------------
def item_pop(train_df, items, i2i):
    cnt = train_df.groupby("item_id").size().reindex(items, fill_value=0)
    pop = np.zeros(len(items), dtype=np.float32)
    for it, c in cnt.items():
        if it in i2i:
            pop[i2i[it]] = float(c)
    return pop


def rerank_pop_penalty(cand_idx, cand_scores, pop_penalty, K, alpha):
    """Subtract a popularity penalty from candidate scores, then take top-K."""
    adj = cand_scores - alpha * pop_penalty[cand_idx]
    order = np.argsort(-adj)[:K]
    return cand_idx[order]


def rerank_fair_prefix(cand_idx, protected_set, K, min_frac):
    """FA*IR-style prefix quota: at every prefix t, #protected >= ceil(min_frac*t).
    Candidates assumed sorted by descending base score.
    """
    K = min(int(K), len(cand_idx))
    prot = [int(x) for x in cand_idx if int(x) in protected_set]
    nonp = [int(x) for x in cand_idx if int(x) not in protected_set]
    out, p, n, taken = [], 0, 0, 0
    for t in range(1, K + 1):
        need = int(math.ceil(min_frac * t))
        if taken < need and p < len(prot):
            out.append(prot[p]); p += 1; taken += 1
        elif n < len(nonp):
            out.append(nonp[n]); n += 1
        elif p < len(prot):
            out.append(prot[p]); p += 1; taken += 1
    return np.array(out, dtype=np.int64)


def rerank_mmr(cand_idx_t, cand_score_t, item_factors, K, mmr_lambda=0.5):
    """Maximal Marginal Relevance re-ranking for diversity."""
    C = int(cand_idx_t.numel())
    K = min(int(K), C)
    lam = float(np.clip(mmr_lambda, 0.0, 1.0))
    V = F.normalize(item_factors[cand_idx_t], dim=1)
    selected = torch.zeros(C, dtype=torch.bool, device=cand_idx_t.device)
    first = torch.argmax(cand_score_t)
    out = [cand_idx_t[first]]
    selected[first] = True
    max_sim = (V @ V[first]).clone()
    for _ in range(1, K):
        mmr = lam * cand_score_t - (1 - lam) * max_sim
        mmr = mmr.masked_fill(selected, float("-inf"))
        j = torch.argmax(mmr)
        out.append(cand_idx_t[j])
        selected[j] = True
        max_sim = torch.maximum(max_sim, V @ V[j])
    return torch.stack(out)
