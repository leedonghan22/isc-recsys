"""S5 -- latent-dimension sensitivity of the baseline gap.

For each latent dimension d, trains a fresh biased MF and measures the baseline
Recall@K gap (no ISC). Shows the gap persists across d (structural, not a
low-capacity artifact) and that RMSE rises with d in the unregularized setting.
Runs across the datasets selected with --dataset (default: all).
"""
import pandas as pd

import _common as C
from src import pipeline
from src.metrics import evaluate
import torch


def run(cfg, out_dir, d_values):
    rows = []
    for d in d_values:
        cfg.latent_factors = d
        p = pipeline.prepare(cfg)
        ones = torch.ones(len(p.users), device=p.device)
        for K in cfg.k_values:
            m = evaluate(p.mf, p.users, p.u2i, p.items, p.i2i, p.groups, ones,
                         p.test, K, cfg.relevance_threshold)
            rows.append({"dataset": cfg.dataset, "d": d, "K": K,
                         "recall_gap": m["recall_gap"], "abs_gap": abs(m["recall_gap"]),
                         "rmse": m["rmse"]})
        print(f"  [{cfg.dataset}] d={d} done")
    return pd.DataFrame(rows)


def main():
    ap = C.base_parser("S5: latent-dimension sweep")
    ap.set_defaults(dataset="all")
    ap.add_argument("--d_values", default="5,8,16,32,64,128")
    args = ap.parse_args()
    d_values = [int(x) for x in args.d_values.split(",")]
    allrows = [run(C.make_config(ds, args), args.out_dir, d_values)
               for ds in C.datasets_from_args(args)]
    C.save_csv(pd.concat(allrows, ignore_index=True), args.out_dir, "s5_latent_sweep.csv")


if __name__ == "__main__":
    main()
