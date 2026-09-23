"""Evaluation metrics: RMSE, Recall@K / NDCG@K per group, and the signed gap.

The ranking evaluation scores every item for each user (full ranking, no
negative sampling) and compares the top-K set against that user's relevant
items (rating >= relevance_threshold) on the held-out fold.
"""
import numpy as np
import torch


def _rmse(y, yhat):
    return torch.sqrt(torch.mean((y - yhat) ** 2))


def _dcg_from_hits(hit01):
    denom = np.log2(np.arange(2, len(hit01) + 2))
    return float(np.sum(hit01 / denom))


def _idcg(rel_len, K):
    L = min(rel_len, K)
    if L <= 0:
        return 0.0
    return float(np.sum(1.0 / np.log2(np.arange(2, L + 2))))


@torch.no_grad()
def evaluate(model, users, u2i, items, i2i, groups, scales, eval_df, K,
             relevance_threshold=3):
    """Return a dict of metrics at cutoff K.

    Keys: rmse, recall_h, recall_g, recall_gap, ndcg_h, ndcg_g, ndcg_gap.
    Gaps are harsh - generous. Users with no relevant items are skipped.
    """
    dev = next(model.parameters()).device
    it = torch.arange(len(items), device=dev)

    # RMSE on rated held-out pairs
    te = eval_df[eval_df.user_id.isin(users) & eval_df.item_id.isin(items)].copy()
    u = torch.tensor(te.user_id.map(u2i).values, dtype=torch.long, device=dev)
    i = torch.tensor(te.item_id.map(i2i).values, dtype=torch.long, device=dev)
    r = torch.tensor(te.rating.values, dtype=torch.float32, device=dev)
    rmse = _rmse(r, model(u, i, scales)).item()

    rec = {"harsh": [], "generous": []}
    ndcg = {"harsh": [], "generous": []}
    for uid in users:
        grp = groups.get(uid, "neutral")
        if grp == "neutral":
            continue
        rel = eval_df[(eval_df.user_id == uid)
                      & (eval_df.rating >= relevance_threshold)].item_id.tolist()
        rel_idx = [i2i[m] for m in rel if m in i2i]
        if not rel_idx:
            continue
        scores = model(torch.full((len(items),), u2i[uid], dtype=torch.long, device=dev),
                       it, scales)
        topk = torch.topk(scores, K).indices.tolist()
        rel_set = set(rel_idx)
        hit01 = np.array([1.0 if it_idx in rel_set else 0.0 for it_idx in topk])
        rec[grp].append(hit01.sum() / len(rel_idx))
        idcg = _idcg(len(rel_idx), K)
        ndcg[grp].append(_dcg_from_hits(hit01) / idcg if idcg > 0 else 0.0)

    rh = float(np.mean(rec["harsh"])) if rec["harsh"] else 0.0
    rg = float(np.mean(rec["generous"])) if rec["generous"] else 0.0
    nh = float(np.mean(ndcg["harsh"])) if ndcg["harsh"] else 0.0
    ng = float(np.mean(ndcg["generous"])) if ndcg["generous"] else 0.0
    return {
        "rmse": rmse,
        "recall_h": rh, "recall_g": rg, "recall_gap": rh - rg,
        "ndcg_h": nh, "ndcg_g": ng, "ndcg_gap": nh - ng,
    }


@torch.no_grad()
def var_group(model, df, groups, u2i, i2i, group="generous"):
    """Within-group variance of the interaction scores s_ui (Var_G)."""
    dev = next(model.parameters()).device
    sub = df[df.user_id.map(groups) == group]
    if sub.empty:
        return 0.0
    u = torch.tensor(sub.user_id.map(u2i).values, dtype=torch.long, device=dev)
    i = torch.tensor(sub.item_id.map(i2i).values, dtype=torch.long, device=dev)
    return float(torch.var(model.raw_dot(u, i), unbiased=True).item())


@torch.no_grad()
def boundary_density(model, users, groups, u2i, items, i2i, K, eps=0.25,
                     subsample=None, seed=0):
    """Mean number of items per generous user whose swap point against the
    K-th-ranked item, tilde_gamma = (b_K - b_i)/(s_ui - s_uK), lies in
    [1-eps, 1+eps]. This is D_G in the paper (boundary-faithful definition).
    Also returns the deprecated fraction-in-[0.5,1.5] proxy for reference.
    """
    dev = next(model.parameters()).device
    n = len(items)
    it = torch.arange(n, device=dev)
    bi = model.bi(it).squeeze().detach().cpu().numpy()

    gen = [u for u in users if groups.get(u) == "generous"]
    if not gen:
        return 0.0, 0.0
    if subsample is not None and subsample < len(gen):
        gen = list(np.random.default_rng(seed).choice(gen, size=subsample, replace=False))

    lo, hi = 1.0 - eps, 1.0 + eps
    counts, old = [], []
    for uid in gen:
        s = model.raw_dot(torch.full((n,), u2i[uid], dtype=torch.long, device=dev),
                          it).detach().cpu().numpy()
        f1 = bi + s
        if n <= K:
            continue
        kth = np.argpartition(f1, n - K)[n - K]
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = (bi[kth] - bi) / (s - s[kth])
        tg = tg[np.isfinite(tg)]
        counts.append(int(np.sum((tg >= lo) & (tg <= hi))))
        old.append(float(np.mean((tg >= 0.5) & (tg <= 1.5))))
    return (float(np.mean(counts)) if counts else 0.0,
            float(np.mean(old)) if old else 0.0)
