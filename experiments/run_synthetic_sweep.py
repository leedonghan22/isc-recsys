"""Table 6 / Section S1 -- controlled synthetic skewness sweep (ML-10M base).

Remaps ML-10M ratings to seven skewness targets (variance free), retrains the
same MF, applies ISC at K*, and records the mechanism quantities: within-group
score variance Var_G, curvature A_G, boundary density D_G, the required
deviation max|gamma-1|, iterations, RMSE cost, and gap reduction. Reports
Spearman correlations across all runs.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _common as C
from src import data as D, model as M, isc, pipeline
from src.metrics import evaluate, var_group, boundary_density
from src import synthetic as SY
import torch

TARGET_SKEWS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def run_cell(cfg, train_raw, test, target_skew, seed, K):
    pipeline.set_seed(seed)
    device = M.get_device()
    tr_s, te_s, ach = SY.reshape_to_target_skew(train_raw, test, target_skew, seed=seed)

    groups = SY.quantile_groups(tr_s, fraction=0.10)
    train = D.filter_items(tr_s, cfg.min_item_ratings)
    users, items, u2i, i2i = M.build_index(train)
    mf = M.train_mf(train, users, items, u2i, i2i, cfg, device, progress=False)

    res = isc.tune(mf, users, items, u2i, i2i, groups, val_df=te_s, cfg=cfg, K=K)
    varG = var_group(mf, train, groups, u2i, i2i, "generous")
    sK = isc.compute_sK_raw(mf, users, items, u2i, K)
    aG = isc.compute_ag_cg(mf, train, groups, u2i, i2i, sK, cfg.beta_sig)["generous"][0]
    dG, dG_old = boundary_density(mf, users, groups, u2i, items, i2i, K)

    b, f = res["baseline"], res["final"]
    red = ((abs(b["recall_gap"]) - abs(f["recall_gap"])) / abs(b["recall_gap"]) * 100
           if abs(b["recall_gap"]) > 0 else float("nan"))
    return {
        "S_R": target_skew, "seed": seed, "rating_var": ach["var"],
        "Var_G": varG, "A_G": aG, "D_G": dG, "D_G_oldfrac": dG_old,
        "max_dev": abs(max(res["gamma_H"] - 1, 1 - res["gamma_G"])),
        "iters": res["iters"], "dRMSE_pct": (f["rmse"] - b["rmse"]) / b["rmse"] * 100,
        "gap_reduction_pct": red, "converged": res["converged"],
    }


def main():
    ap = C.base_parser("Table 6 / S1: synthetic skewness sweep")
    ap.set_defaults(dataset="ml-10m")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    cfg = C.make_config("ml-10m", args)
    K = cfg.anchor_k

    train_raw, _, test = D.load_split(cfg, seed=0)  # base data fixed; skew is remapped

    rows = []
    for sr in TARGET_SKEWS:
        for s in range(args.seeds):
            rows.append(run_cell(cfg, train_raw, test, sr, s, K))
            print(f"  S_R={sr:+.1f} seed={s} done")
    runs = pd.DataFrame(rows)
    C.save_csv(runs, args.out_dir, "table6_s1_synthetic_runs.csv")

    # per-level means (variance-sorted, as in Table 6)
    level = runs.groupby("S_R").mean(numeric_only=True).reset_index().sort_values("rating_var")
    C.save_csv(level, args.out_dir, "table6_s1_synthetic_levels.csv")

    # Spearman correlations across all runs
    def rho(a, b):
        r, p = spearmanr(runs[a], runs[b])
        return {"pair": f"{a}->{b}", "rho": r, "p": p}
    corr = pd.DataFrame([
        rho("rating_var", "Var_G"), rho("rating_var", "A_G"),
        rho("Var_G", "D_G"), rho("Var_G", "iters"), rho("Var_G", "max_dev"),
        rho("A_G", "iters"), rho("Var_G", "dRMSE_pct"), rho("D_G", "dRMSE_pct"),
        rho("Var_G", "D_G_oldfrac"),
    ])
    print(corr.to_string(index=False))
    C.save_csv(corr, args.out_dir, "table6_s1_synthetic_spearman.csv")


if __name__ == "__main__":
    main()
