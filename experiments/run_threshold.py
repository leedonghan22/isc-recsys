"""S5 -- group-threshold sensitivity.

Sweeps the (rel_low, rel_high) thresholds that define harsh/generous, re-grouping
users each time on the raw training frame, and reports the gap reduction at each
cutoff. The MF model is trained once and reused across thresholds.
"""
import pandas as pd

import _common as C
from src import pipeline, isc, data as D


DEFAULT_THRESHOLDS = "0.15:0.85,0.20:0.80,0.25:0.75,0.30:0.70,0.35:0.65"


def run(cfg, out_dir, thresholds):
    p = pipeline.prepare(cfg)
    rows = []
    for lo, hi in thresholds:
        groups = D.group_users(p.train_raw, lo, hi)
        sizes = pipeline.group_sizes(groups)
        for K in cfg.k_values:
            res = isc.tune(p.mf, p.users, p.items, p.u2i, p.i2i, groups,
                           val_df=p.test, cfg=cfg, K=K)
            b, f = res["baseline"]["recall_gap"], res["final"]["recall_gap"]
            red = (abs(b) - abs(f)) / abs(b) * 100 if abs(b) > 0 else float("nan")
            rows.append({"dataset": cfg.dataset, "rel_low": lo, "rel_high": hi,
                         "n_harsh": sizes["harsh"], "n_generous": sizes["generous"],
                         "K": K, "reduction_pct": red,
                         "rmse_BL": res["baseline"]["rmse"], "rmse_ISC": res["final"]["rmse"]})
    df = pd.DataFrame(rows)
    print(df[df.K == cfg.anchor_k].to_string(index=False))
    return df


def main():
    ap = C.base_parser("S5: threshold sensitivity")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    args = ap.parse_args()
    thr = []
    for tok in args.thresholds.split(","):
        lo, hi = (float(x) for x in tok.split(":"))
        thr.append((min(lo, hi), max(lo, hi)))
    allrows = [run(C.make_config(ds, args), args.out_dir, thr) for ds in C.datasets_from_args(args)]
    C.save_csv(pd.concat(allrows, ignore_index=True), args.out_dir, "s5_threshold.csv")


if __name__ == "__main__":
    main()
