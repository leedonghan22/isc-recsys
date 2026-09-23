"""Table 4 (matched-cutoff reduction) and Table S4 (single-K transfer matrix).

For each tuning cutoff K* it tunes (gamma_H, gamma_G) once on the test fold,
then evaluates the resulting gap at every cutoff K. The diagonal (K*==K) is the
matched-cutoff reduction of Table 4; the full grid is the transfer matrix S4.
"""
import numpy as np
import pandas as pd

import _common as C
from src import pipeline, isc
from src.metrics import evaluate
import torch


def run(cfg, out_dir):
    p = pipeline.prepare(cfg)
    Ks = cfg.k_values
    dev = p.device

    mask_h = torch.tensor([p.groups.get(u) == "harsh" for u in p.users], device=dev)
    mask_g = torch.tensor([p.groups.get(u) == "generous" for u in p.users], device=dev)

    detail = []
    for k_star in Ks:
        res = isc.tune(p.mf, p.users, p.items, p.u2i, p.i2i, p.groups,
                       val_df=p.test, cfg=cfg, K=k_star)
        scales = torch.ones(len(p.users), device=dev)
        scales[mask_h] = res["gamma_H"]
        scales[mask_g] = res["gamma_G"]
        ones = torch.ones(len(p.users), device=dev)
        for k_eval in Ks:
            base = evaluate(p.mf, p.users, p.u2i, p.items, p.i2i, p.groups, ones,
                            p.test, k_eval, cfg.relevance_threshold)["recall_gap"]
            isc_gap = evaluate(p.mf, p.users, p.u2i, p.items, p.i2i, p.groups, scales,
                               p.test, k_eval, cfg.relevance_threshold)["recall_gap"]
            red = (abs(base) - abs(isc_gap)) / abs(base) * 100 if abs(base) > 0 else float("nan")
            detail.append({"dataset": cfg.dataset, "K_star": k_star, "K_eval": k_eval,
                           "gap_BL": base, "gap_ISC": isc_gap, "reduction_pct": red})

    df = pd.DataFrame(detail)
    matched = df[df.K_star == df.K_eval][["dataset", "K_star", "reduction_pct"]]
    print(f"\n[{cfg.dataset}] matched-cutoff reduction (Table 4):")
    print(matched.to_string(index=False))
    return df


def main():
    args = C.base_parser("Table 4 / S4: cutoff sweep").parse_args()
    allrows = [run(C.make_config(ds, args), args.out_dir) for ds in C.datasets_from_args(args)]
    df = pd.concat(allrows, ignore_index=True)
    C.save_csv(df, args.out_dir, "table4_s4_cutoffs.csv")


if __name__ == "__main__":
    main()
