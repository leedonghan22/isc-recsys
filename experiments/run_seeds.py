"""Figure 1 and the cross-seed reliability studies (Robustness / S3).

The dataset split is fixed once; only the MF initialization/training seed is
varied. Reports per-seed baseline and ISC gap at each cutoff plus the mean and
a t-based 95% CI. Use --n_seeds 20 for the paper's reliability numbers on every
dataset (this driver replaces the earlier 5-seed Books script and the
ML-10M-only CI script).
"""
import math
import numpy as np
import pandas as pd

import _common as C
from src import pipeline, isc, model as M


def _t95(mean, std, n):
    if n <= 1 or not np.isfinite(std):
        return mean, mean
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
         14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093}.get(n - 1, 1.96)
    half = t * std / math.sqrt(n)
    return mean - half, mean + half


def run(cfg, out_dir, n_seeds, dataset_seed):
    # fix the data split once (dataset_seed); vary only the MF training seed
    device = M.get_device()
    prep0 = pipeline.prepare(cfg, seed=dataset_seed, device=device, train=False)

    rows = []
    for s in range(n_seeds):
        pipeline.set_seed(s)
        mf = M.train_mf(prep0.train, prep0.users, prep0.items, prep0.u2i, prep0.i2i,
                        cfg, device, progress=False)
        for K in cfg.k_values:
            res = isc.tune(mf, prep0.users, prep0.items, prep0.u2i, prep0.i2i,
                           prep0.groups, val_df=prep0.test, cfg=cfg, K=K)
            b, f = res["baseline"]["recall_gap"], res["final"]["recall_gap"]
            red = (abs(b) - abs(f)) / abs(b) * 100 if abs(b) > 0 else float("nan")
            rows.append({"dataset": cfg.dataset, "seed": s, "K": K,
                         "baseline_abs_gap": abs(b), "isc_abs_gap": abs(f),
                         "reduction_pct": red, "converged": res["converged"],
                         "gamma_H": res["gamma_H"]})
        print(f"  [{cfg.dataset}] seed {s} done")

    runs = pd.DataFrame(rows)
    summ = []
    for K in sorted(runs.K.unique()):
        sub = runs[runs.K == K]
        n = len(sub)
        for col in ("baseline_abs_gap", "isc_abs_gap", "reduction_pct"):
            m = float(np.mean(sub[col])); sd = float(np.std(sub[col], ddof=1)) if n > 1 else 0.0
            lo, hi = _t95(m, sd, n)
            summ.append({"dataset": cfg.dataset, "K": K, "metric": col,
                         "mean": m, "ci_low": lo, "ci_high": hi, "n": n})
    C.save_csv(runs, out_dir, f"seeds_{cfg.dataset}_runs.csv")
    C.save_csv(pd.DataFrame(summ), out_dir, f"seeds_{cfg.dataset}_summary.csv")
    return runs


def main():
    ap = C.base_parser("Figure 1 / S3: cross-seed reliability")
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--dataset_seed", type=int, default=0)
    args = ap.parse_args()
    for ds in C.datasets_from_args(args):
        run(C.make_config(ds, args), args.out_dir, args.n_seeds, args.dataset_seed)


if __name__ == "__main__":
    main()
