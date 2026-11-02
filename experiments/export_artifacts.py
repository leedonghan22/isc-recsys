"""Export the exact evaluation splits, group assignments, and catalogs (reviewer 2).

    py experiments/export_artifacts.py --dataset all

Writes, per dataset, to results/revision/artifacts/<ds>/ (gzip CSV):
  train_raw.csv.gz   training interactions before the item filter (grouping basis)
  val.csv.gz         validation fold (5 interactions per user)
  test.csv.gz        test fold (5 most recent interactions per user)
  catalog.csv.gz     catalog items (>= 100 training interactions) with their counts
  groups.csv.gz      user_id, relative positivity rho_u, group (harsh/generous/neutral)
  shares.csv         share of in-catalog held-out pairs per group (val and test)
No model is trained.  Trained MF factors are written by
    py experiments/run_revision.py --dataset <ds> --save_factors
"""
import os

import pandas as pd

import _common as C
from src.config import Config
from src import revision as R

OUT = os.path.join(C.RESULTS_DIR, "revision", "artifacts")


def positivity(train_raw):
    item_avg = train_raw.groupby("item_id")["rating"].mean()
    above = train_raw.rating > train_raw.item_id.map(item_avg)
    return above.groupby(train_raw.user_id).mean()


def main():
    a = C.base_parser("Export splits, groups and catalogs").parse_args()
    for ds in C.datasets_from_args(a):
        cfg = Config(dataset=ds)
        p = R.prepare_strict(cfg, split_seed=0)
        d = os.path.join(OUT, ds)
        os.makedirs(d, exist_ok=True)
        cols = ["user_id", "item_id", "rating", "timestamp"]
        for name in ("train_raw", "val", "test"):
            p[name][cols].sort_values(["user_id", "timestamp"]).to_csv(
                os.path.join(d, f"{name}.csv.gz"), index=False, compression="gzip")
        cnt = p["train"].groupby("item_id").size().rename("train_interactions").reset_index()
        cnt.to_csv(os.path.join(d, "catalog.csv.gz"), index=False, compression="gzip")
        rho = positivity(p["train_raw"]).rename("rho").reset_index()
        rho["group"] = rho.user_id.map(lambda u: p["groups"].get(u, "neutral"))
        rho.to_csv(os.path.join(d, "groups.csv.gz"), index=False, compression="gzip")
        rows = []
        for fold in ("val", "test"):
            f = p[fold]
            f = f[f.user_id.isin(p["u2i"]) & f.item_id.isin(p["i2i"])]
            g = f.user_id.map(lambda u: p["groups"].get(u, "neutral"))
            rows.append({"fold": fold, "pairs": len(f), "share_harsh": (g == "harsh").mean(),
                         "share_generous": (g == "generous").mean()})
        pd.DataFrame(rows).to_csv(os.path.join(d, "shares.csv"), index=False)
        print(f"[saved] {d}")


if __name__ == "__main__":
    main()
