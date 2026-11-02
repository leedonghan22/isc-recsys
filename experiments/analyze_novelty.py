"""Second-round analyses of the revision outputs (CSV only; seconds).

    py experiments/analyze_novelty.py

Reads results/revision/rev_<ds>_isc.csv and rev_<ds>_gamma_acc.csv and writes
results/revision/novelty.md, novelty_*.csv, Fig1.pdf and Fig2.pdf:

  A. Group-level NDCG@K before/after ISC (reviewer 3).
  B. Validation detectability gate: gated ISC (calibrate only if the validation
     gap's bootstrap CI excludes 0, else gamma = 1) versus ungated ISC, overall,
     per dataset, and stratified by cutoff; harmful runs (|gap| widened).
  C. Validation-only forecast of the test accuracy cost.  For each run,
        dMSE_g = A_g (g-1)^2 + 2 (A_g - B_g)(g-1)          (exact identity)
     is evaluated with A_g, B_g measured on the VALIDATION fold and the tuned
     factor, aggregated with the group shares w_g of held-out rated pairs, and
     converted to a relative RMSE change using the validation RMSE.  It is
     compared with the observed test RMSE change.  w_g are recovered exactly
     from the identity on the test fold (least squares; residual ~1e-7), which
     equals the share of in-catalog held-out pairs of each group.
  Fig1: test |gap| vs K, baseline / ISC / gated ISC.
  Fig2: forecast vs observed test RMSE change.
"""
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

import _common as C

OUT = os.path.join(C.RESULTS_DIR, "revision")
DS = ["ml-10m", "ml-1m", "books"]
NAMES = {"ml-10m": "ML-10M", "ml-1m": "ML-1M", "books": "Books"}


def t95(x):
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, m, m
    tc = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
          8: 2.306, 9: 2.262}.get(n - 1, 1.96)
    h = tc * float(x.std(ddof=1)) / math.sqrt(n)
    return m, m - h, m + h


def fci(x, nd=4):
    m, lo, hi = t95(x)
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    isc = pd.concat([pd.read_csv(os.path.join(OUT, f"rev_{d}_isc.csv")) for d in DS], ignore_index=True)
    acc = pd.concat([pd.read_csv(os.path.join(OUT, f"rev_{d}_gamma_acc.csv")) for d in DS], ignore_index=True)
    acc = acc.set_index(["dataset", "model", "seed", "fold", "group"])
    sec = []

    # ---------------- A. group NDCG ----------------
    mf = isc[isc.model == "mf"].copy()
    rows = []
    for (ds, K), s in mf.groupby(["dataset", "K"], sort=False):
        nb, ni = s.test_base_ndcg_gap.abs(), s.test_isc_ndcg_gap.abs()
        rows.append({"Dataset": NAMES[ds], "K": K,
                     "NDCG harsh": f"{s.test_base_ndcg_h.mean():.4f} -> {s.test_isc_ndcg_h.mean():.4f}",
                     "NDCG generous": f"{s.test_base_ndcg_g.mean():.4f} -> {s.test_isc_ndcg_g.mean():.4f}",
                     "dNDCG harsh": fci(s.test_isc_ndcg_h - s.test_base_ndcg_h),
                     "dNDCG generous": fci(s.test_isc_ndcg_g - s.test_base_ndcg_g),
                     "|NDCG gap| base -> ISC": f"{nb.mean():.4f} -> {ni.mean():.4f}",
                     "NDCG-gap reduction": f"{100 * (1 - ni.mean() / nb.mean()):.0f}%"})
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(OUT, "novelty_group_ndcg.csv"), index=False)
    sec.append(("A. Group-level NDCG@K, test fold (base MF, mean over seeds, 95% CI)", A))

    # ---------------- B. detectability gate ----------------
    mf["b"] = mf.test_base_recall_gap.abs()
    mf["i"] = mf.test_isc_recall_gap.abs()
    mf["gated"] = np.where(mf.val_measurable, mf.i, mf.b)
    mf["d"] = mf.b - mf.i
    mf["rm_isc"] = 100 * (mf.test_rmse_isc / mf.test_rmse_base - 1)
    mf["rm_gt"] = np.where(mf.val_measurable, mf.rm_isc, 0.0)
    red = lambda b, x: f"{100 * (1 - x.sum() / b.sum()):.0f}%"
    rows = []
    for ds, s in list(mf.groupby("dataset", sort=False)) + [("all", mf)]:
        rows.append({"Dataset": NAMES.get(ds, "All"), "Runs": len(s),
                     "Calibrated by gate": f"{int(s.val_measurable.sum())}/{len(s)}",
                     "Pooled red. ungated": red(s.b, s.i), "Pooled red. gated": red(s.b, s["gated"]),
                     "Runs widened ungated": f"{int((s.i > s.b).sum())}/{len(s)}",
                     "Runs widened gated": f"{int((s["gated"] > s.b).sum())}/{len(s)}",
                     "Mean dRMSE% ungated": f"{s.rm_isc.mean():+.2f}",
                     "Mean dRMSE% gated": f"{s.rm_gt.mean():+.2f}"})
    B1 = pd.DataFrame(rows)
    rows = []
    for band, s in (("K = 5", mf[mf.K == 5]), ("K = 10", mf[mf.K == 10]), ("K = 20", mf[mf.K == 20]),
                    ("K >= 20", mf[mf.K >= 20]), ("K >= 50", mf[mf.K >= 50])):
        for flag, lab in ((True, "detectable"), (False, "not detectable")):
            t = s[s.val_measurable == flag]
            if len(t):
                rows.append({"Cutoffs": band, "Validation gap": lab, "Runs": len(t),
                             "Pooled red.": red(t.b, t.i), "Improved": f"{int((t.d > 0).sum())}/{len(t)}"})
    B2 = pd.DataFrame(rows)
    harm = mf[mf.i > mf.b]
    note = (f"Runs in which ungated ISC widened the test gap: {len(harm)}; of these, "
            f"{int((~harm.val_measurable).sum())} had a non-detectable validation gap.")
    B1.to_csv(os.path.join(OUT, "novelty_gate.csv"), index=False)
    B2.to_csv(os.path.join(OUT, "novelty_gate_by_K.csv"), index=False)
    sec.append(("B1. Detectability gate (base MF, all cutoffs and seeds)", B1))
    sec.append(("B2. Transfer by validation detectability within cutoff bands (datasets pooled). " + note, B2))

    # ---------------- C. validation-only cost forecast ----------------
    rows = []
    for r in isc.itertuples():
        rec = {"dataset": r.dataset, "model": r.model, "seed": r.seed, "K": r.K,
               "obs_dmse": r.test_rmse_isc ** 2 - r.test_rmse_base ** 2,
               "obs_rel": r.test_rmse_isc / r.test_rmse_base - 1, "val_rmse": r.val_rmse_base}
        for g, gam in (("harsh", r.gamma_H), ("generous", r.gamma_G)):
            for fold in ("val", "test"):
                a, b = acc.loc[(r.dataset, r.model, r.seed, fold, g), ["A_g", "B_g"]]
                dl = gam - 1.0
                rec[f"{fold}_{g}"] = a * dl * dl + 2 * dl * (a - b)
                rec[f"{fold}2_{g}"] = a * dl * dl
        rows.append(rec)
    P = pd.DataFrame(rows)
    W = {}
    for ds in DS:
        s = P[P.dataset == ds]
        X, y = s[["test_harsh", "test_generous"]].values, s.obs_dmse.values
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        W[ds] = (w, float(np.abs(X @ w - y).max()))
    for kind, pre in (("full", "val_"), ("second", "val2_")):
        tot = np.array([W[d][0][0] * h + W[d][0][1] * g for d, h, g in
                        zip(P.dataset, P[pre + "harsh"], P[pre + "generous"])])
        P[f"pred_rel_{kind}"] = np.sqrt(P.val_rmse ** 2 + tot) / P.val_rmse - 1
    P.to_csv(os.path.join(OUT, "novelty_cost_forecast_runs.csv"), index=False)
    rows = []
    for (ds, m), s in list(P.groupby(["dataset", "model"], sort=False)) + [(("all", "all"), P)]:
        ob, pf, p2 = 100 * s.obs_rel, 100 * s.pred_rel_full, 100 * s.pred_rel_second
        rows.append({"Dataset": NAMES.get(ds, "All"), "Model": m, "Runs": len(s),
                     "Observed dRMSE% (mean)": f"{ob.mean():+.2f}",
                     "Forecast, Eq. 5 (mean)": f"{pf.mean():+.2f}",
                     "Forecast, 2nd order only (mean)": f"{p2.mean():+.2f}",
                     "Pearson r (Eq. 5)": f"{stats.pearsonr(pf, ob)[0]:.3f}",
                     "MAE pp (Eq. 5)": f"{np.abs(pf - ob).mean():.3f}",
                     "MAE pp (2nd only)": f"{np.abs(p2 - ob).mean():.3f}"})
    Cc = pd.DataFrame(rows)
    Cc.to_csv(os.path.join(OUT, "novelty_cost_forecast.csv"), index=False)
    wtxt = "; ".join(f"{NAMES[d]}: w_H={W[d][0][0]:.4f}, w_G={W[d][0][1]:.4f} (max residual {W[d][1]:.1e})" for d in DS)
    rows = []
    for ds in DS:
        s = P[P.dataset == ds]
        for g in ("harsh", "generous"):
            rows.append({"Dataset": NAMES[ds], "Group": g,
                         "r(val forecast, test exact)": f"{stats.pearsonr(s['val_' + g], s['test_' + g])[0]:.3f}",
                         "Sign agreement": f"{(np.sign(s['val_' + g]) == np.sign(s['test_' + g])).mean():.2f}"})
    Cg = pd.DataFrame(rows)
    sec.append(("C1. Validation-only forecast of the test RMSE change (all runs). Group shares: " + wtxt, Cc))
    sec.append(("C2. Group-level forecast of the test MSE change", Cg))

    with open(os.path.join(OUT, "novelty.md"), "w", encoding="utf-8") as f:
        for t, df in sec:
            f.write(f"## {t}\n\n{md(df)}\n\n")
    print(f"[saved] {os.path.join(OUT, 'novelty.md')}")

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, ds in zip(axes, DS):
        s = mf[mf.dataset == ds]
        Ks = sorted(s.K.unique())
        for col, lab, c, ls in (("b", "Baseline MF", "#c0392b", "-"), ("i", "ISC (val-tuned)", "#2471a3", "-"),
                                ("gated", "Gated ISC", "#1e8449", "--")):
            st = [t95(s[s.K == K][col]) for K in Ks]
            ax.plot(Ks, [x[0] for x in st], marker="o", ms=3.5, ls=ls, color=c, label=lab)
            if col != "gated":
                ax.fill_between(Ks, [max(x[1], 0) for x in st], [x[2] for x in st], color=c, alpha=0.13, lw=0)
        ax.set_title(NAMES[ds])
        ax.set_xlabel("$K$")
        ax.set_xscale("log")
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_ylabel(r"$|\Delta|$ on test")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig1.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    mk = {"ml-10m": ("o", "#2471a3"), "ml-1m": ("s", "#1e8449"), "books": ("^", "#c0392b")}
    for ax, sub, ttl in ((axes[0], P[P.dataset != "books"], "MovieLens"), (axes[1], P[P.dataset == "books"], "Books")):
        for ds, s in sub.groupby("dataset"):
            ax.scatter(100 * s.pred_rel_full, 100 * s.obs_rel, s=14, marker=mk[ds][0], color=mk[ds][1],
                       alpha=0.8, label=NAMES[ds])
        lim = [min(0, 100 * sub[["pred_rel_full", "obs_rel"]].values.min()) - 0.05,
               100 * sub[["pred_rel_full", "obs_rel"]].values.max() * 1.05 + 0.05]
        ax.plot(lim, lim, color="0.5", lw=0.8, ls=":")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Forecast from validation data (% RMSE)")
        ax.set_ylabel("Observed on test (% RMSE)")
        ax.set_title(ttl)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig2.pdf"))
    plt.close(fig)
    print("[saved] Fig1.pdf, Fig2.pdf")


if __name__ == "__main__":
    main()
