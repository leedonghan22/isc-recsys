"""Revision driver: ONE run per dataset produces every number the revised
manuscript needs (leakage-free protocol, see src/revision.py).

    python experiments/run_revision.py --dataset ml-1m --quick   # 5-10 min smoke test
    python experiments/run_revision.py --dataset all             # full run

Outputs (results/revision/):
  rev_<ds>_isc.csv        per model x seed x K: val baseline gap + bootstrap CI +
                          measurability flag, tuned gammas, convergence/cap flags,
                          test baseline / ISC / ISC-with-fallback metrics
                          (group + overall Recall/NDCG, RMSE, gap CIs)
  rev_<ds>_grid.csv       validation grid search over (gamma_H, gamma_G) at K=20
  rev_<ds>_diag.csv       item-bias-only and popularity rankers (gap mechanism)
  rev_<ds>_groups.csv     data-only group descriptives (activity, relevant counts,
                          popularity of relevant items)
  rev_<ds>_spread.csv     within-user spread of s_ui vs spread of b_i, per group
  rev_<ds>_gamma_acc.csv  A_g, B_g, gamma_acc = B_g/A_g on train / val / test
  rev_<ds>_regselect.csv  validation RMSE of the regularised-MF candidates
  rev_<ds>_data.csv       split and catalog accounting (Table 1)
  rev_ml-10m_ks.csv       KS / Anderson-Darling: 10k-user sample vs full ML-10M
  rev_<ds>_config.json    every hyper-parameter used
"""
import json
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


def scales_for(prep, gH, gG, device):
    """Per-user gamma vector over the catalog index (neutral users keep 1)."""
    if "_gidx" not in prep:
        g = np.array([prep["groups"].get(u, "neutral") for u in prep["users"]])
        prep["_gidx"] = (torch.tensor(g == "harsh"), torch.tensor(g == "generous"))
    h, gm = prep["_gidx"]
    s = torch.ones(len(prep["users"]))
    s[h], s[gm] = gH, gG
    return s.to(device)


def strip(d, prefix):
    return {f"{prefix}_{k}": v for k, v in d.items() if not k.startswith("_")}


def ks_ml10m(cfg):
    from scipy.stats import ks_2samp, anderson_ksamp
    df = pd.read_csv(os.path.join(cfg.data_path(), "ratings.dat"), sep="::",
                     engine="python", names=["user_id", "item_id", "rating", "timestamp"])
    cut = df.user_id.unique()[cfg.num_users() - 1]
    samp = df[df.user_id <= cut]
    rows = []
    for name, f in (("ratings_per_user", lambda x: x.groupby("user_id").size()),
                    ("mean_rating_per_user", lambda x: x.groupby("user_id").rating.mean())):
        a, b = f(samp).values, f(df).values
        ks = ks_2samp(a, b)
        ad = anderson_ksamp([a, b])
        rows.append({"variable": name, "n_sample_users": len(a), "n_full_users": len(b),
                     "KS_D": ks.statistic, "KS_p": ks.pvalue,
                     "AD_stat": ad.statistic, "AD_p_capped": ad.pvalue})
    rows.append({"variable": "rating_value", "n_sample_users": len(samp), "n_full_users": len(df),
                 "KS_D": ks_2samp(samp.rating.values, df.rating.values).statistic,
                 "KS_p": float("nan"), "AD_stat": float("nan"), "AD_p_capped": float("nan")})
    rows.append({"variable": "sampling_rule",
                 "n_sample_users": f"first {cfg.num_users()} distinct user_ids in file order (user_id <= {cut})"})
    return pd.DataFrame(rows)


def run_dataset(ds, a):
    cfg = Config(dataset=ds)
    if a.epochs:
        cfg.epochs_mf = a.epochs
    if a.quick:
        cfg.epochs_mf = 2
    device = torch.device(a.device) if a.device else M.get_device()
    Ks = [cfg.anchor_k] if a.quick else list(cfg.k_values)
    seeds = list(range(1 if a.quick else a.seeds))
    reg_grid = [0.01] if a.quick else a.reg_grid
    d_grid = [cfg.latent_factors] if a.quick else a.d_grid
    grid = np.geomspace(1 / a.grid_max, a.grid_max, 5 if a.quick else a.grid_n)
    B = 200 if a.quick else a.bootstrap
    thr = cfg.relevance_threshold
    tag = f"rev_{ds}"
    os.makedirs(a.out_dir, exist_ok=True)
    save = lambda df, name: C.save_csv(pd.DataFrame(df), a.out_dir, f"{tag}_{name}.csv")

    need = os.path.join(cfg.data_path(), cfg.data_file() or "ratings.dat")
    if not os.path.isfile(need):
        raise SystemExit(f"\n[data missing] {need}\n"
                         f"Put the {ds} file there, or fix the path in configs/datasets.yaml "
                         f"(see data/README.md).\n")
    log(f"{ds}: loading and splitting (device={device})")
    prep = R.prepare_strict(cfg, split_seed=a.split_seed)
    gs = pd.Series({u: prep["groups"].get(u, "neutral") for u in prep["users"]}).value_counts()
    save([{
        "users_in_catalog_index": len(prep["users"]), "items_in_catalog": len(prep["items"]),
        "train_rows_raw": len(prep["train_raw"]), "train_rows_after_item_filter": len(prep["train"]),
        "val_rows": len(prep["val"]), "test_rows": len(prep["test"]),
        "min_item_ratings_in_train": cfg.min_item_ratings,
        "harsh": int(gs.get("harsh", 0)), "generous": int(gs.get("generous", 0)),
        "neutral": int(gs.get("neutral", 0)), "mask_seen": not a.no_mask_seen,
    }], "data")
    save(R.group_descriptives(prep, prep["test"], thr), "groups")
    if ds == "ml-10m" and not a.skip_ks and not a.quick:
        log("ml-10m: KS / AD representativeness of the 10k-user sample")
        save(ks_ml10m(cfg), "ks")

    # ---- regularised MF: pick (reg, d) on validation RMSE, seed 0 ----
    log(f"{ds}: selecting regularised MF on validation RMSE")
    sel, sel_models = [], {}
    for d in d_grid:
        for reg in reg_grid:
            mf = R.train_mf(prep, cfg, device, seed=0, reg=reg, latent_factors=d)
            v = R.rmse_on(mf, prep, prep["val"])
            sel.append({"reg": reg, "d": d, "val_rmse": v})
            sel_models[(reg, d)] = mf
            log(f"   reg={reg} d={d} val RMSE={v:.4f}")
    save(sel, "regselect")
    best = min(sel, key=lambda r: r["val_rmse"])
    models = {"mf": (0.0, cfg.latent_factors, Ks),
              "mf_reg": (best["reg"], best["d"], [cfg.anchor_k])}

    isc_rows, grid_rows, diag_rows, spread_rows, acc_rows = [], [], [], [], []
    for mname, (reg, d, Ks_m) in models.items():
        for seed in seeds:
            if mname == "mf_reg" and seed == 0:
                mf = sel_models[(reg, d)]
            else:
                mf = R.train_mf(prep, cfg, device, seed=seed, reg=reg, latent_factors=d)
            log(f"{ds}: {mname} seed {seed} trained")
            mask = not a.no_mask_seen
            rv = R.Ranker(mf, prep, prep["val"], [prep["train_raw"]], thr, "groups", mask)
            rt = R.Ranker(mf, prep, prep["test"], [prep["train_raw"], prep["val"]], thr, "all", mask)
            ones = scales_for(prep, 1.0, 1.0, device)
            rmse_val0 = R.rmse_on(mf, prep, prep["val"], ones)
            rmse_test0 = R.rmse_on(mf, prep, prep["test"], ones)

            for fold, df in (("train", prep["train"]), ("val", prep["val"]), ("test", prep["test"])):
                for g, (A, Bg, ga) in R.gamma_acc(mf, prep, df).items():
                    acc_rows.append({"dataset": ds, "model": mname, "seed": seed, "fold": fold, "group": g,
                                     "A_g": A, "B_g": Bg, "gamma_acc": ga})
            for g, v in R.score_spread(mf, prep, rt).items():
                spread_rows.append({"dataset": ds, "model": mname, "seed": seed, "group": g, **v})

            for K in Ks_m:
                vb = rv.summary(1.0, 1.0, K)
                lo, hi = R.bootstrap_gap_ci(vb["_rec"], rv.group, B, seed)
                tu = R.tune_isc(rv, K, cfg, a.gamma_cap)
                tb = rt.summary(1.0, 1.0, K)
                ti = rt.summary(tu["gamma_H"], tu["gamma_G"], K)
                tf = tb if tu["failed"] else ti
                blo, bhi = R.bootstrap_gap_ci(tb["_rec"], rt.group, B, seed)
                ilo, ihi = R.bootstrap_gap_ci(ti["_rec"], rt.group, B, seed)
                rmse_isc = R.rmse_on(mf, prep, prep["test"],
                                     scales_for(prep, tu["gamma_H"], tu["gamma_G"], device))
                isc_rows.append({
                    "dataset": ds, "model": mname, "reg": reg, "d": d, "seed": seed, "K": K,
                    "val_gap_baseline": vb["recall_gap"], "val_gap_ci_low": lo, "val_gap_ci_high": hi,
                    "val_measurable": bool(lo > 0 or hi < 0),
                    **{k: v for k, v in tu.items() if k != "val_gap_baseline"},
                    **strip(tb, "test_base"), **strip(ti, "test_isc"), **strip(tf, "test_isc_fallback"),
                    "test_base_gap_ci_low": blo, "test_base_gap_ci_high": bhi,
                    "test_isc_gap_ci_low": ilo, "test_isc_gap_ci_high": ihi,
                    "val_rmse_base": rmse_val0, "test_rmse_base": rmse_test0,
                    "test_rmse_isc": rmse_isc,
                    "test_rmse_isc_fallback": rmse_test0 if tu["failed"] else rmse_isc,
                })
                log(f"   K={K}: val gap {vb['recall_gap']:+.4f} [{lo:+.4f},{hi:+.4f}] -> "
                    f"gamma=({tu['gamma_H']:.3f},{tu['gamma_G']:.3f}) "
                    f"{'FAILED' if tu['failed'] else 'ok'}; test gap "
                    f"{tb['recall_gap']:+.4f} -> {ti['recall_gap']:+.4f}")

            # reviewer-3 comparator: exhaustive validation grid at the anchor cutoff
            K = cfg.anchor_k
            gr = R.tune_grid(rv, K, grid)
            tg = rt.summary(gr["gamma_H"], gr["gamma_G"], K)
            grid_rows.append({"dataset": ds, "model": mname, "seed": seed, "K": K,
                              "grid_points": len(grid) ** 2, **gr, **strip(tg, "test_grid"),
                              "test_rmse_grid": R.rmse_on(mf, prep, prep["test"],
                                    scales_for(prep, gr["gamma_H"], gr["gamma_G"], device))})

            # gap-mechanism diagnostics on test (mf only)
            if mname == "mf":
                for K in Ks:
                    ib = rt.summary(0.0, 0.0, K, gamma_other=0.0)
                    diag_rows.append({"dataset": ds, "seed": seed, "K": K,
                                      "ranker": "item_bias_only", **strip(ib, "test")})
                    if seed == 0:
                        pp = rt.summary(1.0, 1.0, K, item_score=R.popularity_scores(prep, device))
                        diag_rows.append({"dataset": ds, "seed": seed, "K": K,
                                          "ranker": "popularity", **strip(pp, "test")})

    save(isc_rows, "isc")
    save(grid_rows, "grid")
    save(diag_rows, "diag")
    save(spread_rows, "spread")
    save(acc_rows, "gamma_acc")
    meta = cfg.to_dict()
    meta.update({"revision_args": vars(a), "selected_reg_mf": best,
                 "protocol": "MF and groups on train only; gamma tuned on val; test evaluated once",
                 "isc_step": cfg.gamma_lr * (1 + cfg.lam_mult)})
    with open(os.path.join(a.out_dir, f"{tag}_config.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log(f"{ds}: done")


def main():
    p = C.base_parser("Revision: leakage-free ISC evaluation")
    p.set_defaults(out_dir=OUT_DEFAULT)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--gamma_cap", type=float, default=5.0)
    p.add_argument("--reg_grid", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    p.add_argument("--d_grid", type=int, nargs="+", default=[5, 32])
    p.add_argument("--grid_n", type=int, default=17)
    p.add_argument("--grid_max", type=float, default=4.0)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--no_mask_seen", action="store_true",
                   help="rank already-observed items too (original protocol)")
    p.add_argument("--skip_ks", action="store_true")
    p.add_argument("--quick", action="store_true", help="1 seed, 2 epochs, K=20 only")
    p.add_argument("--epochs", type=int, default=None, help="override MF epochs (default 50)")
    p.add_argument("--device", default=None, help="cpu / cuda / mps (default: auto)")
    a = p.parse_args()
    for ds in C.datasets_from_args(a):
        run_dataset(ds, a)


if __name__ == "__main__":
    main()
