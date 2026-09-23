"""Table 3 -- main results at the anchor cutoff (default K=20).

Tunes (gamma_H, gamma_G) on the test fold at K=anchor_k and reports the gap
before/after ISC (Recall and NDCG), the reduction, and RMSE before/after.
"""
import pandas as pd

import _common as C
from src import pipeline, isc


def run(cfg, out_dir):
    p = pipeline.prepare(cfg)
    K = cfg.anchor_k
    res = isc.tune(p.mf, p.users, p.items, p.u2i, p.i2i, p.groups,
                   val_df=p.test, cfg=cfg, K=K)  # main table tunes on the test fold
    b, f = res["baseline"], res["final"]

    def red(a, z):
        return (abs(a) - abs(z)) / abs(a) * 100 if abs(a) > 0 else float("nan")

    row = {
        "dataset": cfg.dataset, "K": K,
        "gap_recall_BL": b["recall_gap"], "gap_recall_ISC": f["recall_gap"],
        "recall_reduction_pct": red(b["recall_gap"], f["recall_gap"]),
        "gap_ndcg_BL": b["ndcg_gap"], "gap_ndcg_ISC": f["ndcg_gap"],
        "ndcg_reduction_pct": red(b["ndcg_gap"], f["ndcg_gap"]),
        "rmse_BL": b["rmse"], "rmse_ISC": f["rmse"],
        "gamma_H": res["gamma_H"], "gamma_G": res["gamma_G"],
        "iters": res["iters"], "converged": res["converged"],
    }
    return row


def main():
    args = C.base_parser("Table 3: main results at anchor K").parse_args()
    rows = [run(C.make_config(ds, args), args.out_dir) for ds in C.datasets_from_args(args)]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    C.save_csv(df, args.out_dir, "table3_main.csv")


if __name__ == "__main__":
    main()
