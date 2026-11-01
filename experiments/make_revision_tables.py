"""Turn results/revision/*.csv into manuscript-ready tables (+ one figure).

    python experiments/make_revision_tables.py

Writes results/revision/tables.md (all tables, readable), one .tex snippet per
table, and fig_gap_vs_K.pdf (replacement for Figure 1). Runs in seconds.
"""
import glob
import os

import numpy as np
import pandas as pd

import _common as C
from src.revision import t95

OUT = os.path.join(C.RESULTS_DIR, "revision")
ORDER = ["ml-10m", "ml-1m", "books"]
NAMES = {"ml-10m": "ML-10M", "ml-1m": "ML-1M", "books": "Books"}


def load(kind):
    fs = sorted(glob.glob(os.path.join(OUT, f"rev_*_{kind}.csv")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def ci(x, fmt="{:.4f}", nonneg=True):
    m, lo, hi = t95(x)
    lo = max(lo, 0.0) if nonneg else lo
    return f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def red(base, new):
    b, n = np.nanmean(np.abs(base)), np.nanmean(np.abs(new))
    return f"{100 * (1 - n / b):.0f}%" if b > 0 else "–"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def tex(df, caption):
    cols = list(df.columns)
    esc = lambda s: str(s).replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
    body = [" & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    body += [" & ".join(esc(v) for v in r) + r" \\" for r in df.itertuples(index=False)]
    return ("\\begin{table}[h]\n\\centering\n\\caption{" + caption + "}\n\\small\n"
            "\\begin{tabular}{" + "l" * len(cols) + "}\n\\toprule\n" + "\n".join(body)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    isc, grid, diag = load("isc"), load("grid"), load("diag")
    groups_frames = []
    for f in sorted(glob.glob(os.path.join(OUT, "rev_*_groups.csv"))):
        g = pd.read_csv(f)
        g.insert(0, "dataset", os.path.basename(f)[4:-11])
        groups_frames.append(g)
    groups = pd.concat(groups_frames, ignore_index=True) if groups_frames else pd.DataFrame()
    spread, acc = load("spread"), load("gamma_acc")
    tables = []

    mf = isc[isc.model == "mf"]
    ds_list = [d for d in ORDER if d in set(mf.dataset)]

    # ---- T1: main leakage-free result -------------------------------------
    rows = []
    for ds in ds_list:
        for K in sorted(mf[mf.dataset == ds].K.unique()):
            s = mf[(mf.dataset == ds) & (mf.K == K)]
            rows.append({
                "Dataset": NAMES[ds], "K": K,
                "Val-measurable": f"{int(s.val_measurable.sum())}/{len(s)}",
                "|Δ| baseline (test)": ci(s.test_base_recall_gap.abs()),
                "|Δ| ISC (test)": ci(s.test_isc_recall_gap.abs()),
                "Reduction": red(s.test_base_recall_gap, s.test_isc_recall_gap),
                "Reduction (fallback)": red(s.test_base_recall_gap, s.test_isc_fallback_recall_gap),
                "Failed tunings": f"{int(s.failed.sum())}/{len(s)}",
            })
    tables.append(("T1", "Leakage-free validation-to-test results (mean over seeds, 95% CI). "
                   "Fallback: a failed tuning reverts to gamma=1.", pd.DataFrame(rows)))

    # ---- T2: absolute group / overall utility -----------------------------
    rows = []
    for ds in ds_list:
        for K in sorted(mf[mf.dataset == ds].K.unique()):
            s = mf[(mf.dataset == ds) & (mf.K == K)]
            f = lambda c: f"{s['test_base_' + c].mean():.4f} → {s['test_isc_' + c].mean():.4f}"
            rows.append({"Dataset": NAMES[ds], "K": K,
                         "Recall harsh": f("recall_h"), "Recall generous": f("recall_g"),
                         "Recall overall": f("recall_all"), "NDCG overall": f("ndcg_all"),
                         "RMSE": f"{s.test_rmse_base.mean():.4f} → {s.test_rmse_isc.mean():.4f}"})
    tables.append(("T2", "Absolute test utility before → after ISC (mean over seeds).",
                   pd.DataFrame(rows)))

    # ---- T3: regularised MF + grid comparator at K=20 ---------------------
    rows = []
    for ds in ds_list:
        for mname, label in (("mf", "MF (unregularised)"), ("mf_reg", "MF (L2, val-tuned)")):
            s = isc[(isc.dataset == ds) & (isc.model == mname) & (isc.K == 20)]
            gsub = grid[(grid.dataset == ds) & (grid.model == mname)] if not grid.empty else grid
            if s.empty:
                continue
            rows.append({
                "Dataset": NAMES[ds], "Base model": label,
                "reg / d": f"{s.reg.iloc[0]} / {s.d.iloc[0]}",
                "RMSE base": f"{s.test_rmse_base.mean():.4f}",
                "|Δ| base": f"{s.test_base_recall_gap.abs().mean():.4f}",
                "|Δ| ISC": f"{s.test_isc_recall_gap.abs().mean():.4f}",
                "|Δ| grid": f"{gsub.test_grid_recall_gap.abs().mean():.4f}" if len(gsub) else "–",
                "Recall overall base/ISC/grid": (f"{s.test_base_recall_all.mean():.4f} / "
                                                 f"{s.test_isc_recall_all.mean():.4f} / "
                                                 + (f"{gsub.test_grid_recall_all.mean():.4f}" if len(gsub) else "–")),
                "RMSE ISC / grid": (f"{s.test_rmse_isc.mean():.4f} / "
                                    + (f"{gsub.test_rmse_grid.mean():.4f}" if len(gsub) else "–")),
            })
    tables.append(("T3", "K=20: regularised base model and validation grid-search comparator "
                   "(mean over seeds).", pd.DataFrame(rows)))

    # ---- T4: gap mechanism diagnostics ------------------------------------
    if not diag.empty:
        rows = []
        for ds in ds_list:
            for K in sorted(diag[diag.dataset == ds].K.unique()):
                b = mf[(mf.dataset == ds) & (mf.K == K)]
                ib = diag[(diag.dataset == ds) & (diag.K == K) & (diag.ranker == "item_bias_only")]
                pp = diag[(diag.dataset == ds) & (diag.K == K) & (diag.ranker == "popularity")]
                rows.append({"Dataset": NAMES[ds], "K": K,
                             "Δ MF": f"{b.test_base_recall_gap.mean():+.4f}" if len(b) else "–",
                             "Δ item-bias only": f"{ib.test_recall_gap.mean():+.4f}",
                             "Δ popularity": f"{pp.test_recall_gap.mean():+.4f}" if len(pp) else "–"})
        tables.append(("T4", "Signed test gap (harsh − generous) of the MF baseline vs. "
                       "rankers that ignore personalisation.", pd.DataFrame(rows)))
    if not groups.empty:
        g = groups[groups.group != "neutral"].copy()
        g["dataset"] = g.dataset.map(NAMES)
        tables.append(("T5", "Group descriptives (test fold, data only).", g.round(3)))
    if not spread.empty:
        sp = spread[spread.model == "mf"]
        tables.append(("T6", "Within-user spread of s_ui vs. spread of b_i (mean over seeds).",
                       sp.groupby(["dataset", "group"])[["mean_std_s", "std_b_i", "ratio"]]
                       .mean().round(4).reset_index()))
    if not acc.empty:
        tables.append(("T7", "gamma_acc = B_g/A_g by fold (mean over seeds). "
                       "≈1 on train is the stationarity condition.",
                       acc.groupby(["dataset", "model", "fold", "group"])[["A_g", "B_g", "gamma_acc"]]
                       .mean().round(4).reset_index()))

    # ---- T8: per-seed outcomes incl. failures (R3-V4) ---------------------
    cols = ["dataset", "model", "seed", "K", "gamma_H", "gamma_G", "iters", "converged",
            "capped", "test_base_recall_gap", "test_isc_recall_gap", "test_rmse_base",
            "test_rmse_isc"]
    per_seed = isc[isc.K == 20][cols].copy()
    per_seed[cols[4:6] + cols[9:]] = per_seed[cols[4:6] + cols[9:]].round(4)
    tables.append(("T8", "Per-seed outcomes at K=20, including failed tunings.", per_seed))

    # ---- write -----------------------------------------------------------
    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as fmd:
        for name, cap, df in tables:
            fmd.write(f"## {name}. {cap}\n\n{md(df)}\n\n")
            df.to_csv(os.path.join(OUT, f"table_{name}.csv"), index=False)
            with open(os.path.join(OUT, f"table_{name}.tex"), "w", encoding="utf-8") as ft:
                ft.write(tex(df, cap))
    print(f"[saved] {os.path.join(OUT, 'tables.md')}")

    # ---- figure: test |Δ| vs K with 95% CI (replaces Figure 1) -------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(ds_list), figsize=(4 * len(ds_list), 3.2), squeeze=False)
        for ax, ds in zip(axes[0], ds_list):
            s = mf[mf.dataset == ds]
            Ks = sorted(s.K.unique())
            for col, lab, c in (("test_base_recall_gap", "Baseline MF", "tab:red"),
                                ("test_isc_recall_gap", "ISC (val-tuned)", "tab:blue")):
                st = [t95(s[s.K == K][col].abs()) for K in Ks]
                m = [x[0] for x in st]
                ax.plot(Ks, m, "o-", color=c, label=lab)
                ax.fill_between(Ks, [x[1] for x in st], [x[2] for x in st], color=c, alpha=0.15)
            ax.set_title(NAMES[ds]); ax.set_xlabel("K"); ax.set_ylabel("|Δ| on test")
        axes[0][0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_gap_vs_K.pdf"))
        print(f"[saved] {os.path.join(OUT, 'fig_gap_vs_K.pdf')}")
    except Exception as e:  # figure is optional
        print(f"[figure skipped] {e}")


if __name__ == "__main__":
    main()
