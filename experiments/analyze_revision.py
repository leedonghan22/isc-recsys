"""Seed-level analysis of the revision outputs. Reads the CSVs only (seconds);
no model is trained.

    py experiments/analyze_revision.py

Writes results/revision/analysis.md with
  A. paired seed-level gap reduction per dataset x K
     (d_s = |gap_base,s| - |gap_ISC,s|; mean, 95% t-CI, seeds improved)
  B. transfer conditioned on whether the VALIDATION baseline gap was
     statistically detectable (user-level bootstrap CI excludes 0)
  C. where parity comes from: change in harsh / generous / overall Recall@K
  D. accuracy-cost decomposition on the test fold, per group:
       MSE_g(gamma) - MSE_g(1) = A_g (gamma-1)^2  +  2 (gamma-1)(A_g - B_g)
                                 second order        first order
     with A_g = E[s^2], B_g = E[(r - mu - b_u - b_i) s] measured on test.
     This is an exact identity, so it shows how much of the RMSE cost is the
     first-order term that appears when gamma_acc = B_g/A_g != 1 on held-out data.
"""
import glob
import os

import numpy as np
import pandas as pd

import _common as C
from src.revision import t95

OUT = os.path.join(C.RESULTS_DIR, "revision")
ORDER = {"ml-10m": 0, "ml-1m": 1, "books": 2}
NAMES = {"ml-10m": "ML-10M", "ml-1m": "ML-1M", "books": "Books"}


def load(kind):
    fs = sorted(glob.glob(os.path.join(OUT, f"rev_*_{kind}.csv")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def fmt_ci(x, nd=4):
    m, lo, hi = t95(x)
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def pooled_red(b, i):
    sb = np.abs(b).sum()
    return f"{100 * (1 - np.abs(i).sum() / sb):.0f}%" if sb > 0 else "–"


def main():
    isc, acc = load("isc"), load("gamma_acc")
    if isc.empty:
        raise SystemExit("no results/revision/rev_*_isc.csv found - run run_revision.py first")
    isc = isc.sort_values(["dataset", "model", "K", "seed"],
                          key=lambda s: s.map(ORDER) if s.name == "dataset" else s)
    isc["d"] = isc.test_base_recall_gap.abs() - isc.test_isc_recall_gap.abs()
    isc["d_fb"] = isc.test_base_recall_gap.abs() - isc.test_isc_fallback_recall_gap.abs()
    sections = []

    # ---- A. paired seed-level reduction ----------------------------------
    rows = []
    for (ds, m, K), s in isc.groupby(["dataset", "model", "K"], sort=False):
        lo = t95(s.d)[1]
        rows.append({"Dataset": NAMES[ds], "Model": m, "K": K, "n": len(s),
                     "|Δ| base − |Δ| ISC (95% CI)": fmt_ci(s.d),
                     "CI > 0": "yes" if lo > 0 else "no",
                     "Seeds improved": f"{int((s.d > 0).sum())}/{len(s)}",
                     "Same, with fallback": fmt_ci(s.d_fb)})
    sections.append(("A. Paired seed-level gap reduction on test "
                     "(positive = ISC narrowed the gap)", pd.DataFrame(rows)))

    # ---- B. conditioned on validation detectability ----------------------
    rows = []
    mf = isc[isc.model == "mf"]
    for ds, s in list(mf.groupby("dataset", sort=False)) + [("all", mf)]:
        for flag, lab in ((True, "detectable on val"), (False, "not detectable on val")):
            t = s[s.val_measurable == flag]
            if t.empty:
                continue
            rows.append({"Dataset": NAMES.get(ds, "All"), "Validation gap": lab,
                         "Runs (seed×K)": len(t),
                         "Pooled reduction": pooled_red(t.test_base_recall_gap, t.test_isc_recall_gap),
                         "Runs improved": f"{int((t.d > 0).sum())}/{len(t)}",
                         "Mean d (95% CI)": fmt_ci(t.d),
                         "Failed tunings": f"{int(t.failed.sum())}/{len(t)}"})
    sections.append(("B. Out-of-sample transfer by validation detectability "
                     "(MF, all K pooled)", pd.DataFrame(rows)))

    # ---- C. who moves ------------------------------------------------------
    rows = []
    for (ds, K), s in mf.groupby(["dataset", "K"], sort=False):
        rows.append({"Dataset": NAMES[ds], "K": K,
                     "Δ Recall harsh": fmt_ci(s.test_isc_recall_h - s.test_base_recall_h),
                     "Δ Recall generous": fmt_ci(s.test_isc_recall_g - s.test_base_recall_g),
                     "Δ Recall overall": fmt_ci(s.test_isc_recall_all - s.test_base_recall_all),
                     "Δ NDCG overall": fmt_ci(s.test_isc_ndcg_all - s.test_base_ndcg_all),
                     "Δ RMSE (%)": fmt_ci(100 * (s.test_rmse_isc / s.test_rmse_base - 1), 2)})
    sections.append(("C. Change after ISC on test (MF; mean over seeds, 95% CI)",
                     pd.DataFrame(rows)))

    # ---- D. accuracy-cost decomposition ------------------------------------
    if not acc.empty:
        te = acc[acc.fold == "test"]
        rows = []
        for r in isc.itertuples():
            for grp, gam in (("harsh", r.gamma_H), ("generous", r.gamma_G)):
                a = te[(te.dataset == r.dataset) & (te.model == r.model)
                       & (te.seed == r.seed) & (te.group == grp)]
                if a.empty:
                    continue
                A, B = float(a.A_g.iloc[0]), float(a.B_g.iloc[0])
                dlt = gam - 1.0
                rows.append({"dataset": r.dataset, "model": r.model, "K": r.K, "group": grp,
                             "gamma": gam, "gamma_acc": B / A if A > 0 else np.nan,
                             "second_order": A * dlt ** 2, "first_order": 2 * dlt * (A - B)})
        dec = pd.DataFrame(rows)
        if not dec.empty:
            dec["total"] = dec["second_order"] + dec["first_order"]
            summ = (dec.groupby(["dataset", "model", "K", "group"], sort=False)
                    [["gamma", "gamma_acc", "second_order", "first_order", "total"]].mean().reset_index())
            summ["first-order share"] = (summ["first_order"].abs()
                                         / (summ["first_order"].abs() + summ["second_order"].abs())).map(
                lambda v: f"{100 * v:.0f}%" if np.isfinite(v) else "–")
            summ["dataset"] = summ.dataset.map(NAMES)
            summ = summ[summ.K == 20].round(4)
            summ = summ.rename(columns={"second_order": "ΔMSE 2nd-order", "first_order": "ΔMSE 1st-order",
                                        "total": "ΔMSE total"})
            sections.append(("D. Accuracy-cost decomposition on test at K=20 "
                             "(per group, mean over seeds)", summ))

    path = os.path.join(OUT, "analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        for title, df in sections:
            f.write(f"## {title}\n\n{md(df)}\n\n")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
