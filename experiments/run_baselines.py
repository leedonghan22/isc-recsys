"""Section S2 -- comparison with post-processing baselines (ML-10M).

Monotonic calibrators (isotonic, linear spline) preserve item order, so the
user-side gap is unchanged while RMSE may improve. Re-rankers (MMR, popularity
penalty, FA*IR-style prefix) change item order for other objectives; their gap
impact is erratic across cutoffs. ISC is reported alongside for reference.
"""
import numpy as np
import pandas as pd

import _common as C
from src import pipeline, isc, baselines as B
from src.metrics import evaluate
import torch


def _rmse_np(pred, r):
    return float(np.sqrt(np.mean((pred - r) ** 2)))


def _group_gap_from_topk(topk_fn, p, K):
    """Compute harsh-generous Recall@K gap where topk_fn(uid)->list of item idx."""
    rec = {"harsh": [], "generous": []}
    for uid in p.users:
        grp = p.groups.get(uid, "neutral")
        if grp == "neutral":
            continue
        rel = p.test[(p.test.user_id == uid) & (p.test.rating >= 3)].item_id.tolist()
        rel_idx = set(p.i2i[m] for m in rel if m in p.i2i)
        if not rel_idx:
            continue
        topk = topk_fn(uid)
        rec[grp].append(len(set(topk) & rel_idx) / len(rel_idx))
    rh = np.mean(rec["harsh"]) if rec["harsh"] else 0.0
    rg = np.mean(rec["generous"]) if rec["generous"] else 0.0
    return rh - rg


def run(cfg, out_dir, cand=200):
    p = pipeline.prepare(cfg)
    dev = p.device
    it_all = torch.arange(len(p.items), device=dev)
    ones = torch.ones(len(p.users), device=dev)
    Ks = cfg.k_values

    # cached vanilla per-user scores (top-`cand` candidates)
    cand_cache = {}
    with torch.no_grad():
        for uid in p.users:
            sc = p.mf(torch.full((len(p.items),), p.u2i[uid], dtype=torch.long, device=dev),
                      it_all, None)
            top = torch.topk(sc, min(cand, len(p.items)))
            cand_cache[uid] = (top.indices, top.values)

    rows = []
    base_gap = {K: evaluate(p.mf, p.users, p.u2i, p.items, p.i2i, p.groups, ones,
                            p.test, K, cfg.relevance_threshold)["recall_gap"] for K in Ks}

    # ---- monotonic calibrators: RMSE changes, ranking/gap unchanged ----
    tr = p.train
    s_tr = p.mf(torch.tensor(tr.user_id.map(p.u2i).values, dtype=torch.long, device=dev),
                torch.tensor(tr.item_id.map(p.i2i).values, dtype=torch.long, device=dev),
                None).detach().cpu().numpy()
    r_tr = tr.rating.values
    te = p.test[p.test.user_id.isin(p.users) & p.test.item_id.isin(p.items)]
    s_te = p.mf(torch.tensor(te.user_id.map(p.u2i).values, dtype=torch.long, device=dev),
                torch.tensor(te.item_id.map(p.i2i).values, dtype=torch.long, device=dev),
                None).detach().cpu().numpy()
    r_te = te.rating.values

    for name, fit, app in [
        ("Isotonic", B.fit_isotonic, B.apply_isotonic),
        ("LinearSpline", B.fit_piecewise_linear, B.apply_piecewise_linear),
    ]:
        try:
            m = fit(s_tr, r_tr)
            pred = app(m, s_te) if name == "Isotonic" else app(m[0], m[1], s_te)
            rmse_c = _rmse_np(pred, r_te)
        except Exception as e:
            print(f"[warn] {name} skipped: {e}")
            continue
        for K in Ks:
            rows.append({"dataset": cfg.dataset, "method": name, "K": K,
                         "gap": base_gap[K], "reduction_pct": 0.0, "rmse": rmse_c})

    # ---- re-rankers ----
    pop = B.item_pop(p.train, p.items, p.i2i)
    tail = set(np.where(pop <= np.quantile(pop.astype(float), 0.8))[0].tolist())
    pop_pen = np.log1p(pop)
    Q = p.mf.Q.weight.detach()

    for K in Ks:
        # MMR
        def mmr_topk(uid, K=K):
            ci, cs = cand_cache[uid]
            return B.rerank_mmr(ci, cs, Q, K, mmr_lambda=0.5).tolist()
        # popularity penalty
        def pop_topk(uid, K=K):
            ci, cs = cand_cache[uid]
            return B.rerank_pop_penalty(ci.cpu().numpy(), cs.cpu().numpy(), pop_pen, K,
                                        alpha=1.0).tolist()
        # FA*IR prefix (protect tail items)
        def fair_topk(uid, K=K):
            ci, _ = cand_cache[uid]
            return B.rerank_fair_prefix(ci.cpu().numpy(), tail, K, min_frac=0.3).tolist()

        for name, fn in [("MMR", mmr_topk), ("PopPenalty", pop_topk), ("FAIR", fair_topk)]:
            gap = _group_gap_from_topk(fn, p, K)
            red = (abs(base_gap[K]) - abs(gap)) / abs(base_gap[K]) * 100 if abs(base_gap[K]) > 0 else float("nan")
            rows.append({"dataset": cfg.dataset, "method": name, "K": K,
                         "gap": gap, "reduction_pct": red, "rmse": float("nan")})

    # ---- ISC for reference ----
    for K in Ks:
        res = isc.tune(p.mf, p.users, p.items, p.u2i, p.i2i, p.groups,
                       val_df=p.test, cfg=cfg, K=K)
        b, f = res["baseline"]["recall_gap"], res["final"]["recall_gap"]
        red = (abs(b) - abs(f)) / abs(b) * 100 if abs(b) > 0 else float("nan")
        rows.append({"dataset": cfg.dataset, "method": "ISC", "K": K,
                     "gap": f, "reduction_pct": red, "rmse": res["final"]["rmse"]})

    df = pd.DataFrame(rows)
    print(df.pivot_table(index="method", columns="K", values="reduction_pct").to_string())
    return df


def main():
    ap = C.base_parser("S2: post-processing baselines")
    ap.set_defaults(dataset="ml-10m")
    args = ap.parse_args()
    allrows = [run(C.make_config(ds, args), args.out_dir) for ds in C.datasets_from_args(args)]
    C.save_csv(pd.concat(allrows, ignore_index=True), args.out_dir, "s2_baselines.csv")


if __name__ == "__main__":
    main()
