"""Regenerate the evaluation folds, group assignments, catalogs and rating
statistics, and write SHA-256 checksums of the folds.

    py experiments/export_artifacts.py --dataset all            # export
    py experiments/export_artifacts.py --dataset all --verify   # check against release/

Per dataset, two folders are written under results/revision/artifacts/<ds>/:

  release/   files that can be archived (identifiers and statistics, no ratings)
    catalog.csv          catalog item ids (>= 100 training interactions) and their counts
    groups.csv           user_id, relative positivity rho_u, group (harsh/generous/neutral),
                         in_catalog_index (True if the user has an in-catalog training
                         interaction; the paper's group counts refer to these users)
    shares.csv           share of in-catalog held-out rated pairs per group (val and test)
    rating_stats.csv     users, ratings, mean, variance, skewness, share of 5-star ratings
    checksums.sha256     SHA-256 of the canonical CSV of train_raw, val and test

  local/     the folds themselves (they contain ratings: keep them local)
    train_raw.csv.gz, val.csv.gz, test.csv.gz

The MovieLens licence does not permit redistribution of the ratings, and the
Amazon ratings are not redistributed either, so only release/ is archived.
A reader who downloads the original files regenerates local/ with this script;
--verify recomputes the checksums and compares them with release/.

The canonical CSV of a fold has the columns user_id,item_id,rating,timestamp,
rows sorted by (user_id, timestamp, item_id), no index, and "\\n" line endings,
so the checksums do not depend on the operating system or on gzip metadata.

No model is trained.  Trained MF factors are written by
    py experiments/run_revision.py --dataset <ds> --save_factors
"""
import gzip
import hashlib
import os

import numpy as np
import pandas as pd

import _common as C
from src.config import Config
from src import revision as R

OUT = os.path.join(C.RESULTS_DIR, "revision", "artifacts")
COLS = ["user_id", "item_id", "rating", "timestamp"]
FOLDS = ("train_raw", "val", "test")


def positivity(train_raw):
    item_avg = train_raw.groupby("item_id")["rating"].mean()
    above = train_raw.rating > train_raw.item_id.map(item_avg)
    return above.groupby(train_raw.user_id).mean()


def canonical_csv(df):
    d = df[COLS].sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    return d.to_csv(index=False, lineterminator="\n")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rating_stats(ds, p):
    r = pd.concat([p[f] for f in FOLDS])["rating"].to_numpy(dtype=float)
    m = r.mean()
    m2 = ((r - m) ** 2).mean()
    m3 = ((r - m) ** 3).mean()
    return pd.DataFrame([{
        "dataset": ds,
        "users": pd.concat([p[f] for f in FOLDS]).user_id.nunique(),
        "ratings": len(r),
        "mean": round(m, 4),
        "variance": round(m2, 4),
        "skewness": round(m3 / m2 ** 1.5, 4),       # population (moment) skewness
        "share_5_star": round(float(np.mean(r == 5.0)), 4),
    }])


def export(ds, p, rel, loc):
    sums = []
    for name in FOLDS:
        text = canonical_csv(p[name])
        sums.append(f"{sha256(text)}  {name}.csv")
        with open(os.path.join(loc, f"{name}.csv.gz"), "wb") as fh:
            with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
                gz.write(text.encode("utf-8"))
    with open(os.path.join(rel, "checksums.sha256"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(sums) + "\n")

    cnt = p["train"].groupby("item_id").size().rename("train_interactions").reset_index()
    cnt.sort_values("item_id").to_csv(os.path.join(rel, "catalog.csv"), index=False, lineterminator="\n")
    rho = positivity(p["train_raw"]).rename("rho").reset_index()
    rho["group"] = rho.user_id.map(lambda u: p["groups"].get(u, "neutral"))
    # users with at least one in-catalog training interaction; only these are scored
    # and evaluated, and the group counts reported in the paper refer to them
    rho["in_catalog_index"] = rho.user_id.isin(p["u2i"])
    rho.sort_values("user_id").to_csv(os.path.join(rel, "groups.csv"), index=False, lineterminator="\n")

    rows = []
    for fold in ("val", "test"):
        f = p[fold]
        f = f[f.user_id.isin(p["u2i"]) & f.item_id.isin(p["i2i"])]
        g = f.user_id.map(lambda u: p["groups"].get(u, "neutral"))
        rows.append({"fold": fold, "pairs": len(f), "share_harsh": (g == "harsh").mean(),
                     "share_generous": (g == "generous").mean()})
    pd.DataFrame(rows).to_csv(os.path.join(rel, "shares.csv"), index=False, lineterminator="\n")
    rating_stats(ds, p).to_csv(os.path.join(rel, "rating_stats.csv"), index=False, lineterminator="\n")


def verify(ds, p, rel):
    path = os.path.join(rel, "checksums.sha256")
    if not os.path.exists(path):
        print(f"[{ds}] no checksum file at {path}")
        return False
    with open(path, encoding="utf-8") as fh:
        want = dict(line.split()[::-1] for line in fh if line.strip())
    ok = True
    for name in FOLDS:
        got = sha256(canonical_csv(p[name]))
        match = want.get(f"{name}.csv") == got
        ok &= match
        print(f"[{ds}] {name:9s} {'OK' if match else 'MISMATCH'}")
    return ok


def main():
    ap = C.base_parser("Export folds, groups, catalogs, rating statistics and checksums")
    ap.add_argument("--verify", action="store_true",
                    help="recompute the fold checksums and compare them with release/")
    a = ap.parse_args()
    all_ok = True
    for ds in C.datasets_from_args(a):
        p = R.prepare_strict(Config(dataset=ds), split_seed=0)
        rel = os.path.join(OUT, ds, "release")
        loc = os.path.join(OUT, ds, "local")
        if a.verify:
            all_ok &= verify(ds, p, rel)
            continue
        os.makedirs(rel, exist_ok=True)
        os.makedirs(loc, exist_ok=True)
        export(ds, p, rel, loc)
        print(f"[saved] {rel} (archivable) and {loc} (local only)")
    if a.verify:
        print("all folds verified" if all_ok else "verification FAILED")


if __name__ == "__main__":
    main()
