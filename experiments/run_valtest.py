"""Table 7 -- same-fold vs strict validation-to-test transfer.

Uses a 3-way temporal split (test = last k, val = previous k, train = rest).
MF and grouping use train (+val for grouping). "val->test" tunes gamma on the
validation fold and evaluates on the unseen test fold; "same-fold" tunes and
evaluates on the test fold. Reported at the cutoffs where the gap is measurable.
"""
import pandas as pd

import _common as C
from src import pipeline, isc


def run(cfg, out_dir, cutoffs=(50, 100)):
    p = pipeline.prepare(cfg, three_way=True)
    rows = []
    for K in cutoffs:
        # strict: tune on val, evaluate on test
        strict = isc.tune(p.mf, p.users, p.items, p.u2i, p.i2i, p.groups,
                          val_df=p.val, cfg=cfg, K=K, eval_df=p.test)
        # same-fold: tune and evaluate on test
        same = isc.tune(p.mf, p.users, p.items, p.u2i, p.i2i, p.groups,
                        val_df=p.test, cfg=cfg, K=K)

        def red(r):
            b, f = r["baseline"]["recall_gap"], r["final"]["recall_gap"]
            return (abs(b) - abs(f)) / abs(b) * 100 if abs(b) > 0 else float("nan")

        rows.append({"dataset": cfg.dataset, "K": K,
                     "same_fold_reduction_pct": red(same),
                     "val_to_test_reduction_pct": red(strict),
                     "val_to_test_converged": strict["converged"]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def main():
    args = C.base_parser("Table 7: validation-to-test transfer").parse_args()
    allrows = [run(C.make_config(ds, args), args.out_dir) for ds in C.datasets_from_args(args)]
    df = pd.concat(allrows, ignore_index=True)
    C.save_csv(df, args.out_dir, "table7_valtest.csv")


if __name__ == "__main__":
    main()
