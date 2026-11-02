"""Ablation of score-scaling strategies (reviewer 1, Q2), leakage-free protocol.

    py experiments/run_ablation.py --dataset ml-10m            # 5 seeds, K=20

For each MF seed (same training as run_revision.py, unregularised MF) and K=20:
  isc          : interaction term scaled per group, sign update on validation
  bias_scaled  : item bias scaled by 1/gamma_g with the ISC factors.  Rank-
                 equivalent to ISC (Lemma 1) -> identical lists, different RMSE
  whole_score  : whole prediction of each group scaled by gamma_g.  A positive
                 per-user rescaling cannot change any list (Lemma 1)
  user_shift   : user bias of each group shifted by +-1.  Cannot change any list
  global       : one interaction factor for ALL users, chosen on validation by a
                 33-point log grid in [1/4, 4] (no group information)
  harsh_only   : sign update on gamma_H only (gamma_G = 1)
  generous_only: sign update on gamma_G only (gamma_H = 1)
Writes results/revision/rev_<ds>_ablation.csv.
"""
import os
import time

import numpy as np
import pandas as pd
import torch

import _common as C
from src.config import Config
from src import model as M
from src import revision as R

OUT_DEFAULT = os.path.join(C.RESULTS_DIR, "revision")


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


@torch.no_grad()
def per_user_weights(rk, wb_h, wb_g, ws_h, ws_g, K, wb_n=1.0, ws_n=1.0,
                     scale_h=1.0, scale_g=1.0, shift_h=0.0, shift_g=0.0):
    """Per-user Recall@K with score = scale*(mu + b_u + shift + wb*b_i + ws*s_ui)."""
    mf = rk.mf
    it = torch.arange(rk.n_items, device=rk.dev)
    Q, bi = mf.Q(it), mf.bi(it).squeeze(1)
    out = []
    for s, e, rel_keys, seen_keys in rk.chunks:
        ui = rk.uidx[s:e]
        h, g = rk.is_h[s:e], rk.is_g[s:e]
        def vec(a, b, c):
            v = torch.full((e - s,), float(c), device=rk.dev)
            v[h], v[g] = float(a), float(b)
            return v.unsqueeze(1)
        wb, ws = vec(wb_h, wb_g, wb_n), vec(ws_h, ws_g, ws_n)
        sc_ = vec(scale_h, scale_g, 1.0)
        sh = vec(shift_h, shift_g, 0.0)
        sc = sc_ * ((mf.mu + mf.bu(ui).squeeze(1)).unsqueeze(1) + sh
                    + wb * bi.unsqueeze(0) + ws * (mf.P(ui) @ Q.T))
        if seen_keys.numel():
            sc.view(-1)[seen_keys] = float("-inf")
        top = torch.topk(sc, K, dim=1).indices
        keys = top + (torch.arange(e - s, device=rk.dev) * rk.n_items).unsqueeze(1)
        out.append(torch.isin(keys, rel_keys).float().sum(1) / rk.n_rel[s:e])
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def rmse_weights(mf, prep, df, wb, ws, scale, shift):
    """RMSE of scale*(mu + b_u + shift + wb*b_i + ws*s) with per-group (H, G) weights."""
    dev = next(mf.parameters()).device
    te = df[df.user_id.isin(prep["u2i"]) & df.item_id.isin(prep["i2i"])]
    grp = te.user_id.map(lambda u: prep["groups"].get(u, "neutral")).values
    def v(pair, neutral):
        a = np.full(len(te), neutral, dtype=np.float32)
        a[grp == "harsh"], a[grp == "generous"] = pair
        return torch.tensor(a, device=dev)
    u = torch.tensor(te.user_id.map(prep["u2i"]).values, dtype=torch.long, device=dev)
    i = torch.tensor(te.item_id.map(prep["i2i"]).values, dtype=torch.long, device=dev)
    r = torch.tensor(te.rating.values, dtype=torch.float32, device=dev)
    pred = v(scale, 1.0) * (mf.mu + mf.bu(u).squeeze(1) + v(shift, 0.0)
                            + v(wb, 1.0) * mf.bi(i).squeeze(1) + v(ws, 1.0) * mf.raw_dot(u, i))
    return float(torch.sqrt(torch.mean((r - pred) ** 2)))


def one_sided(rv, K, cfg, which, cap=5.0):
    step0 = cfg.gamma_lr * (1.0 + cfg.lam_mult)
    base = rv.gap(1.0, 1.0, K)
    eps = max(abs(base) * cfg.epsilon_ratio, 1e-4)
    g, it, conv = 1.0, 0, False
    for it in range(1, cfg.max_iter + 1):
        gap = rv.gap(g, 1.0, K) if which == "H" else rv.gap(1.0, g, K)
        if abs(gap) <= eps:
            conv = True
            break
        s = 1.0 if gap > 0 else -1.0
        g = g * (1 + step0 * s) if which == "H" else g / (1 + step0 * s)
        if max(g, 1 / g) > cap:
            break
    return g, it, conv


def gap_of(rec, group):
    return float(rec[group == "harsh"].mean() - rec[group == "generous"].mean())


def main():
    p = C.base_parser("Ablation of score-scaling strategies")
    p.set_defaults(out_dir=OUT_DEFAULT)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None, help="override MF epochs (testing only)")
    a = p.parse_args()
    for ds in C.datasets_from_args(a):
        cfg = Config(dataset=ds)
        if a.epochs:
            cfg.epochs_mf = a.epochs
        os.makedirs(a.out_dir, exist_ok=True)
        device = torch.device(a.device) if a.device else M.get_device()
        K = a.K
        prep = R.prepare_strict(cfg, split_seed=0)
        thr = cfg.relevance_threshold
        rows = []
        for seed in range(a.seeds):
            mf = R.train_mf(prep, cfg, device, seed=seed)
            log(f"{ds}: seed {seed} trained")
            rv = R.Ranker(mf, prep, prep["val"], [prep["train_raw"]], thr, "groups", True)
            rt = R.Ranker(mf, prep, prep["test"], [prep["train_raw"], prep["val"]], thr, "all", True)
            base = rt.summary(1.0, 1.0, K)
            rmse0 = R.rmse_on(mf, prep, prep["test"])
            def add(name, rec_test, rmse, gH=np.nan, gG=np.nan, extra=None):
                rows.append({"dataset": ds, "seed": seed, "K": K, "strategy": name,
                             "gamma_H": gH, "gamma_G": gG,
                             "val_gap_baseline": rv.gap(1.0, 1.0, K),
                             "test_gap_base": base["recall_gap"],
                             "test_gap": gap_of(rec_test, rt.group),
                             "recall_all_base": base["recall_all"],
                             "recall_all": float(rec_test.mean()),
                             "rmse_base": rmse0, "rmse": rmse, **(extra or {})})
            # ISC
            tu = R.tune_isc(rv, K, cfg, 5.0)
            gH, gG = tu["gamma_H"], tu["gamma_G"]
            rec_isc = rt.per_user(gH, gG, K)[0]
            add("isc", rec_isc, rmse_weights(mf, prep, prep["test"], (1, 1), (gH, gG), (1, 1), (0, 0)),
                gH, gG, {"iters": tu["iters"], "failed": tu["failed"]})
            # rank-equivalent item-bias scaling
            rec_b = per_user_weights(rt, 1 / gH, 1 / gG, 1.0, 1.0, K)
            add("bias_scaled", rec_b,
                rmse_weights(mf, prep, prep["test"], (1 / gH, 1 / gG), (1, 1), (1, 1), (0, 0)),
                gH, gG, {"lists_identical_to_isc": bool(np.allclose(rec_b, rec_isc))})
            # whole-score scaling and user-bias shift (cannot change lists)
            rec_w = per_user_weights(rt, 1, 1, 1, 1, K, scale_h=gH, scale_g=gG)
            add("whole_score", rec_w,
                rmse_weights(mf, prep, prep["test"], (1, 1), (1, 1), (gH, gG), (0, 0)), gH, gG,
                {"lists_identical_to_base": bool(np.allclose(rec_w, base["_rec"]))})
            rec_s = per_user_weights(rt, 1, 1, 1, 1, K, shift_h=-1.0, shift_g=1.0)
            add("user_shift", rec_s,
                rmse_weights(mf, prep, prep["test"], (1, 1), (1, 1), (1, 1), (-1.0, 1.0)), np.nan, np.nan,
                {"lists_identical_to_base": bool(np.allclose(rec_s, base["_rec"]))})
            # global factor (no groups): validation grid
            grid = np.geomspace(0.25, 4.0, 33)
            best = min(grid, key=lambda g: (abs(gap_of(per_user_weights(rv, 1, 1, g, g, K, ws_n=g), rv.group)), abs(np.log(g))))
            rec_g = per_user_weights(rt, 1, 1, best, best, K, ws_n=best)
            te = prep["test"]
            mfp = mf
            with torch.no_grad():
                d = te[te.user_id.isin(prep["u2i"]) & te.item_id.isin(prep["i2i"])]
                u = torch.tensor(d.user_id.map(prep["u2i"]).values, dtype=torch.long, device=device)
                i = torch.tensor(d.item_id.map(prep["i2i"]).values, dtype=torch.long, device=device)
                r = torch.tensor(d.rating.values, dtype=torch.float32, device=device)
                pr = mfp.mu + mfp.bu(u).squeeze(1) + mfp.bi(i).squeeze(1) + best * mfp.raw_dot(u, i)
                rmse_glob = float(torch.sqrt(torch.mean((r - pr) ** 2)))
            add("global", rec_g, rmse_glob, best, best)
            # one-sided
            for which in ("H", "G"):
                g1, it1, conv1 = one_sided(rv, K, cfg, which)
                gh, gg = (g1, 1.0) if which == "H" else (1.0, g1)
                add("harsh_only" if which == "H" else "generous_only", rt.per_user(gh, gg, K)[0],
                    rmse_weights(mf, prep, prep["test"], (1, 1), (gh, gg), (1, 1), (0, 0)), gh, gg,
                    {"iters": it1, "failed": not conv1})
            log(f"   isc gap {base['recall_gap']:+.4f}->{gap_of(rec_isc, rt.group):+.4f}")
            pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, f"rev_{ds}_ablation.csv"), index=False)
        log(f"{ds}: done")


if __name__ == "__main__":
    main()
