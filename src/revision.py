"""Revision protocol (PeerJ CS resubmission).

Everything the revised manuscript needs comes from this module, driven by
experiments/run_revision.py. The original modules (data, model, isc, metrics)
are reused unchanged, so earlier results stay reproducible.

Protocol changes requested by the reviewers
-------------------------------------------
* Leakage-free split. Temporal 3-way split (test = last 5, val = previous 5,
  train = rest). The base MF is trained on TRAIN ONLY and harsh/generous groups
  are defined on TRAIN ONLY. gamma is tuned on VAL; TEST is evaluated once.
* Already-observed items are excluded from the full ranking (train items when
  ranking for val; train + val items when ranking for test). Switch off with
  mask_seen=False to recover the original protocol.
* Fast vectorised evaluation (batched P Q^T + top-k) returning per-user values,
  so we can bootstrap confidence intervals and report absolute group / overall
  Recall@K and NDCG@K.

The ISC update is numerically identical to src/isc.py: there the step is
gamma_lr * (1 + lambda/lambda_max) with lambda = lam_mult * lambda_max, i.e. the
constant gamma_lr * (1 + lam_mult). lambda_max cancels, so it is not computed.
"""
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import data as D
from . import model as M


# --------------------------------------------------------------------------
# Data preparation (leakage-free)
# --------------------------------------------------------------------------
def prepare_strict(cfg, split_seed=0):
    """3-way temporal split; groups and catalog are built from TRAIN only."""
    train_raw, val, test = D.load_split(cfg, seed=split_seed, three_way=True)
    groups = D.group_users(train_raw, cfg.rel_low, cfg.rel_high)   # train only
    train = D.filter_items(train_raw, cfg.min_item_ratings)
    users, items, u2i, i2i = M.build_index(train)
    return {"train_raw": train_raw, "train": train, "val": val, "test": test,
            "groups": groups, "users": users, "items": items, "u2i": u2i, "i2i": i2i}


# --------------------------------------------------------------------------
# Matrix factorisation (unregularised = original; regularised = new baseline)
# --------------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_mf(prep, cfg, device, seed, reg=0.0, latent_factors=None, progress=False):
    """reg == 0 and latent_factors None -> exactly the original M.train_mf.

    reg > 0 adds an L2 penalty on the parameters touched by each mini-batch:
    loss = MSE + reg * mean(|p_u|^2 + |q_i|^2 + b_u^2 + b_i^2).
    """
    set_seed(seed)
    d = cfg.latent_factors if latent_factors is None else latent_factors
    if reg == 0.0 and d == cfg.latent_factors:
        return M.train_mf(prep["train"], prep["users"], prep["items"],
                          prep["u2i"], prep["i2i"], cfg, device, progress=progress)

    mf = M.BiasedMF(len(prep["users"]), len(prep["items"]), d).to(device)
    opt = torch.optim.Adam(mf.parameters(), lr=cfg.lr_mf)
    loader = DataLoader(M.RatingDS(prep["train"], prep["u2i"], prep["i2i"]),
                        batch_size=cfg.batch_size, shuffle=True)
    for _ in range(cfg.epochs_mf):
        for u, i, r in loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pen = (mf.P(u).pow(2).sum(1) + mf.Q(i).pow(2).sum(1)
                   + mf.bu(u).squeeze(1).pow(2) + mf.bi(i).squeeze(1).pow(2)).mean()
            loss = F.mse_loss(mf(u, i), r) + reg * pen
            opt.zero_grad()
            loss.backward()
            opt.step()
    mf.eval()
    return mf


@torch.no_grad()
def rmse_on(mf, prep, df, scales=None):
    dev = next(mf.parameters()).device
    te = df[df.user_id.isin(prep["u2i"]) & df.item_id.isin(prep["i2i"])]
    if te.empty:
        return float("nan")
    u = torch.tensor(te.user_id.map(prep["u2i"]).values, dtype=torch.long, device=dev)
    i = torch.tensor(te.item_id.map(prep["i2i"]).values, dtype=torch.long, device=dev)
    r = torch.tensor(te.rating.values, dtype=torch.float32, device=dev)
    return float(torch.sqrt(torch.mean((r - mf(u, i, scales)) ** 2)).item())


# --------------------------------------------------------------------------
# Fast ranking evaluator
# --------------------------------------------------------------------------
class Ranker:
    """Full-ranking evaluator for one evaluation fold.

    users_scope: "groups" (harsh + generous only; used inside tuning loops) or
                 "all" (every user; used for overall Recall/NDCG).
    Only users with >= 1 relevant (rating >= thr) in-catalog item in eval_df
    are evaluated, exactly as in src/metrics.evaluate.
    """

    def __init__(self, mf, prep, eval_df, exclude_dfs, rel_thr, users_scope="groups",
                 mask_seen=True, chunk=4096):
        self.mf = mf
        self.dev = next(mf.parameters()).device
        self.n_items = len(prep["items"])
        self.chunk = chunk
        u2i, i2i, groups = prep["u2i"], prep["i2i"], prep["groups"]

        rel = eval_df[(eval_df.rating >= rel_thr) & eval_df.item_id.isin(i2i)
                      & eval_df.user_id.isin(u2i)]
        rel_by_user = rel.groupby("user_id").item_id.apply(lambda s: [i2i[x] for x in s])
        g_of = {u: groups.get(u, "neutral") for u in rel_by_user.index}
        if users_scope == "groups":
            uids = [u for u in rel_by_user.index if g_of[u] in ("harsh", "generous")]
        else:
            uids = list(rel_by_user.index)
        uids = sorted(uids, key=lambda u: u2i[u])
        self.uids = uids
        self.uidx = torch.tensor([u2i[u] for u in uids], dtype=torch.long, device=self.dev)
        self.group = np.array([g_of[u] for u in uids])
        self.is_h = torch.tensor(self.group == "harsh", device=self.dev)
        self.is_g = torch.tensor(self.group == "generous", device=self.dev)
        self.n_rel = torch.tensor([len(set(rel_by_user[u])) for u in uids],
                                  dtype=torch.float32, device=self.dev)

        # seen items (masked out of the ranking)
        seen_pos = {u: [] for u in uids}
        if mask_seen and exclude_dfs:
            ex = pd.concat(exclude_dfs)
            ex = ex[ex.user_id.isin(set(uids)) & ex.item_id.isin(i2i)]
            for u, its in ex.groupby("user_id").item_id:
                seen_pos[u] = [i2i[x] for x in its]

        # per-chunk flat keys (row_local * n_items + item) for relevant and seen items
        self.chunks = []
        for s in range(0, len(uids), chunk):
            cu = uids[s:s + chunk]
            rk, sk = [], []
            for r, u in enumerate(cu):
                base = r * self.n_items
                rk.extend(base + j for j in set(rel_by_user[u]))
                sk.extend(base + j for j in seen_pos[u])
            self.chunks.append((
                s, min(s + chunk, len(uids)),
                torch.tensor(sorted(rk), dtype=torch.long, device=self.dev),
                torch.tensor(sk, dtype=torch.long, device=self.dev)))

    @torch.no_grad()
    def per_user(self, gH, gG, K, gamma_other=1.0, item_score=None):
        """Per-user Recall@K and NDCG@K (numpy arrays aligned with self.uids).

        item_score: optional (n_items,) tensor that replaces the model score
        (used for the popularity ranker diagnostic).
        """
        mf = self.mf
        it = torch.arange(self.n_items, device=self.dev)
        Q = mf.Q(it)
        bi = mf.bi(it).squeeze(1)
        disc = 1.0 / torch.log2(torch.arange(2, K + 2, device=self.dev, dtype=torch.float32))
        cdisc = torch.cumsum(disc, 0)
        rec_all, ndcg_all = [], []
        for s, e, rel_keys, seen_keys in self.chunks:
            ui = self.uidx[s:e]
            if item_score is None:
                gam = torch.full((e - s,), float(gamma_other), device=self.dev)
                gam[self.is_h[s:e]] = float(gH)
                gam[self.is_g[s:e]] = float(gG)
                sc = (mf.mu + mf.bu(ui).squeeze(1)).unsqueeze(1) + bi.unsqueeze(0) \
                    + gam.unsqueeze(1) * (mf.P(ui) @ Q.T)
            else:
                sc = item_score.unsqueeze(0).expand(e - s, -1).clone()
            if seen_keys.numel():
                sc.view(-1)[seen_keys] = float("-inf")
            top = torch.topk(sc, K, dim=1).indices
            keys = top + (torch.arange(e - s, device=self.dev) * self.n_items).unsqueeze(1)
            hits = torch.isin(keys, rel_keys).float()
            nrel = self.n_rel[s:e]
            rec_all.append(hits.sum(1) / nrel)
            idcg = cdisc[(torch.clamp(nrel, max=K) - 1).long()]
            ndcg_all.append((hits * disc).sum(1) / idcg)
        return torch.cat(rec_all).cpu().numpy(), torch.cat(ndcg_all).cpu().numpy()

    def summary(self, gH, gG, K, **kw):
        rec, nd = self.per_user(gH, gG, K, **kw)
        h, g = self.group == "harsh", self.group == "generous"
        m = lambda x, mask: float(x[mask].mean()) if mask.any() else float("nan")
        return {"recall_h": m(rec, h), "recall_g": m(rec, g),
                "recall_gap": m(rec, h) - m(rec, g),
                "ndcg_h": m(nd, h), "ndcg_g": m(nd, g), "ndcg_gap": m(nd, h) - m(nd, g),
                "recall_all": float(rec.mean()), "ndcg_all": float(nd.mean()),
                "n_h": int(h.sum()), "n_g": int(g.sum()), "n_all": len(rec),
                "_rec": rec}

    def gap(self, gH, gG, K):
        rec, _ = self.per_user(gH, gG, K)
        return float(rec[self.group == "harsh"].mean() - rec[self.group == "generous"].mean())


def bootstrap_gap_ci(rec, group, B=1000, seed=0, alpha=0.05):
    """User-level bootstrap CI of the harsh-generous gap (resample within group)."""
    rng = np.random.default_rng(seed)
    rh, rg = rec[group == "harsh"], rec[group == "generous"]
    if len(rh) == 0 or len(rg) == 0:
        return float("nan"), float("nan")
    bh = rh[rng.integers(0, len(rh), (B, len(rh)))].mean(1)
    bg = rg[rng.integers(0, len(rg), (B, len(rg)))].mean(1)
    d = bh - bg
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


# --------------------------------------------------------------------------
# Tuners
# --------------------------------------------------------------------------
def tune_isc(ranker_val, K, cfg, gamma_cap=5.0):
    """Sign-update tuner (same update as src/isc.tune) with a symmetric cap.

    Failure = no convergence within max_iter, or any gamma leaving
    [1/gamma_cap, gamma_cap]. The cap is fixed before seeing any test data.
    """
    step0 = cfg.gamma_lr * (1.0 + cfg.lam_mult)
    base = ranker_val.gap(1.0, 1.0, K)
    eps = max(abs(base) * cfg.epsilon_ratio, 1e-4)
    gH, gG = 1.0, 1.0
    converged, capped, iters = False, False, 0
    for t in range(1, cfg.max_iter + 1):
        iters = t
        gap = ranker_val.gap(gH, gG, K)
        if abs(gap) <= eps:
            converged = True
            break
        step = step0 * (1.0 if gap > 0 else -1.0)
        gH, gG = gH * (1 + step), gG / (1 + step)
        if max(gH, 1 / gH, gG, 1 / gG) > gamma_cap:
            capped = True
            break
    return {"gamma_H": gH, "gamma_G": gG, "iters": iters, "converged": converged,
            "capped": capped, "failed": (not converged) or capped,
            "eps": eps, "val_gap_baseline": base}


def tune_grid(ranker_val, K, grid):
    """Exhaustive validation search over (gamma_H, gamma_G): minimise |gap|,
    ties broken by the higher mean group recall (reviewer 3's comparator)."""
    best = None
    for gH in grid:
        for gG in grid:
            rec, _ = ranker_val.per_user(gH, gG, K)
            h, g = ranker_val.group == "harsh", ranker_val.group == "generous"
            gap = rec[h].mean() - rec[g].mean()
            key = (round(abs(gap), 12), -(rec[h].mean() + rec[g].mean()))
            if best is None or key < best[0]:
                best = (key, gH, gG, gap)
    return {"gamma_H": best[1], "gamma_G": best[2], "val_gap": best[3]}


# --------------------------------------------------------------------------
# Diagnostics for the gap mechanism and the accuracy-cost analysis
# --------------------------------------------------------------------------
@torch.no_grad()
def gamma_acc(mf, prep, df):
    """Per-group A_g = E[s^2], B_g = E[(r - mu - b_u - b_i) s], gamma_acc = B_g/A_g."""
    dev = next(mf.parameters()).device
    out = {}
    d = df[df.user_id.isin(prep["u2i"]) & df.item_id.isin(prep["i2i"])]
    for g in ("harsh", "generous", "neutral"):
        sub = d[d.user_id.map(lambda u: prep["groups"].get(u, "neutral")) == g]
        if sub.empty:
            out[g] = (float("nan"),) * 3
            continue
        u = torch.tensor(sub.user_id.map(prep["u2i"]).values, dtype=torch.long, device=dev)
        i = torch.tensor(sub.item_id.map(prep["i2i"]).values, dtype=torch.long, device=dev)
        r = torch.tensor(sub.rating.values, dtype=torch.float32, device=dev)
        s = mf.raw_dot(u, i)
        resid = r - mf.mu - mf.bu(u).squeeze(1) - mf.bi(i).squeeze(1)
        A = float((s * s).mean())
        B = float((resid * s).mean())
        out[g] = (A, B, B / A if A > 0 else float("nan"))
    return out


@torch.no_grad()
def score_spread(mf, prep, ranker):
    """Per group: mean within-user std of s_ui over the catalog vs std of b_i.
    Their ratio says how personalised (vs item-bias driven) a user's ranking is."""
    it = torch.arange(len(prep["items"]), device=ranker.dev)
    bi_std = float(mf.bi(it).squeeze(1).std())
    s_std = []
    for s, e, _, _ in ranker.chunks:
        S = mf.P(ranker.uidx[s:e]) @ mf.Q(it).T
        s_std.append(S.std(1).cpu().numpy())
    s_std = np.concatenate(s_std)
    out = {}
    for g in ("harsh", "generous"):
        m = ranker.group == g
        out[g] = {"mean_std_s": float(s_std[m].mean()), "std_b_i": bi_std,
                  "ratio": float(s_std[m].mean() / bi_std) if bi_std > 0 else float("nan")}
    return out


def group_descriptives(prep, eval_df, rel_thr):
    """Data-only per-group statistics (no model): activity, relevant-item counts,
    and the training popularity of the relevant held-out items."""
    tr, i2i, groups = prep["train"], prep["i2i"], prep["groups"]
    pop = tr.groupby("item_id").size()
    act = prep["train_raw"].groupby("user_id").size()
    ev = eval_df[eval_df.user_id.isin(prep["u2i"])].copy()
    ev["grp"] = ev.user_id.map(lambda u: groups.get(u, "neutral"))
    ev["rel"] = ev.rating >= rel_thr
    ev["in_cat"] = ev.item_id.isin(i2i)
    rows = []
    for g in ("harsh", "generous", "neutral"):
        sub = ev[ev.grp == g]
        rel_in = sub[sub.rel & sub.in_cat]
        per_user_rel = rel_in.groupby("user_id").size()
        users_g = [u for u in prep["u2i"] if groups.get(u, "neutral") == g]
        rows.append({
            "group": g, "n_users": len(users_g),
            "train_ratings_per_user": float(act.reindex(users_g).mean()),
            "mean_train_rating": float(prep["train_raw"][prep["train_raw"].user_id.isin(users_g)].rating.mean()),
            "share_heldout_relevant": float(sub.rel.mean()) if len(sub) else float("nan"),
            "share_relevant_in_catalog": float(sub[sub.rel].in_cat.mean()) if sub.rel.any() else float("nan"),
            "relevant_items_per_evaluated_user": float(per_user_rel.mean()) if len(per_user_rel) else float("nan"),
            "n_evaluated_users": int(len(per_user_rel)),
            "median_train_popularity_of_relevant": float(rel_in.item_id.map(pop).median()) if len(rel_in) else float("nan"),
            "median_train_popularity_catalog": float(pop.median()),
        })
    return pd.DataFrame(rows)


def popularity_scores(prep, device):
    pop = prep["train"].groupby("item_id").size()
    v = np.array([pop.get(m, 0) for m in prep["items"]], dtype=np.float32)
    return torch.tensor(v, device=device)


def t95(x):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if n == 1:
        return m, m, m
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262}.get(n - 1, 1.96)
    h = tcrit * float(x.std(ddof=1)) / math.sqrt(n)
    return m, m - h, m + h
