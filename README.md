# Interaction-Score Calibration (ISC)

Code to reproduce the experiments in *"Interaction-Score Calibration for Ranking
Parity in Latent-Factor Recommender Systems."*

ISC is a lightweight post-processing calibrator for biased matrix-factorization
recommenders. It rescales only the interaction term of a trained model by a
per-group scalar `gamma_g`, tuned offline by an anti-symmetric sign update, so
it needs no retraining and no demographic data. As a representative application
it equalizes Recall@K between behaviourally defined harsh and generous raters.

## Repository layout

```
isc-recsys/
├── src/                       shared, dataset-agnostic library
│   ├── config.py              hyper-parameters + dataset-path resolution
│   ├── data.py                loaders (ML-1M/10M, Books), grouping, splitting
│   ├── model.py               BiasedMF + training
│   ├── isc.py                 ISC: A_g/C_g, lambda_max, the sign-update tuner
│   ├── metrics.py             Recall@K, NDCG@K, RMSE, gap, Var_G, boundary density
│   ├── baselines.py           S2 baselines (isotonic, spline, MMR, pop, FA*IR)
│   ├── synthetic.py           skew remapping + quantile grouping (S1)
│   └── pipeline.py            shared load -> group -> filter -> train flow
├── experiments/               thin drivers, one per table/figure
├── configs/datasets.yaml      YOU create this (see below)
├── scripts/download_data.sh   dataset download notes
└── results/                   CSV outputs land here
```

## Setup

```bash
pip install -r requirements.txt
cp configs/datasets.example.yaml configs/datasets.yaml
# then edit configs/datasets.yaml to point at your local dataset folders
```

The datasets are public and are **not** redistributed here:

- MovieLens 1M / 10M — https://grouplens.org/datasets/movielens/ (folders with `ratings.dat`)
- Amazon Books 5-core — https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ (`Books_5.json`, one JSON per line)

`bash scripts/download_data.sh` prints these links.

## Reproducing the paper

Every driver reads `configs/datasets.yaml`, runs the experiment, and writes a CSV
to `results/`. `--dataset` selects `ml-10m`, `ml-1m`, `books`, or `all`.

| Paper item | Command |
|---|---|
| Table 3 (main results, K=20) | `python experiments/run_main.py --dataset all` |
| Table 4 + Table S4 (cutoff sweep / transfer matrix) | `python experiments/run_cutoffs.py --dataset all` |
| Table 7 (validation-to-test) | `python experiments/run_valtest.py --dataset all` |
| Table 5 / Table 6 / S1 (synthetic skewness sweep) | `python experiments/run_synthetic_sweep.py --seeds 5` |
| S2 (post-processing baselines) | `python experiments/run_baselines.py --dataset ml-10m` |
| Figure 1 / S3 (cross-seed reliability, 20 seeds) | `python experiments/run_seeds.py --dataset all --n_seeds 20` |
| S5 (threshold sensitivity) | `python experiments/run_threshold.py --dataset all` |
| S5 (latent-dimension sweep) | `python experiments/run_latent_sweep.py --dataset all` |

Notes:

- **Cutoffs / groups.** Evaluation is anchored at K=20 with sweeps over
  {5,10,20,50,100}. Harsh/generous groups use positivity thresholds (0.2, 0.8),
  defined on the raw training frame before the item filter.
- **Reliability studies use 20 seeds** on every dataset (`run_seeds.py`,
  `--n_seeds 20`); this single driver covers the ML-10M, ML-1M, and Books
  reliability numbers. The synthetic sweep uses 5 seeds per skew level
  (`run_synthetic_sweep.py --seeds 5`), as in the paper.
- **Compute.** ISC tuning evaluates full-ranking Recall@K at every iteration, so
  large runs (ML-10M/Books at high K) are compute-heavy; this matches the
  evaluation protocol in the paper. A CUDA or Apple-MPS device is used
  automatically when available.
- **Determinism.** Seeds are fixed per run; exact figures can still vary slightly
  with hardware/BLAS and PyTorch versions.

## Revised evaluation protocol (resubmission)

All numbers in the revised manuscript come from one driver per dataset:

```bash
python experiments/run_revision.py --dataset ml-1m --quick   # smoke test (minutes)
python experiments/run_revision.py --dataset all             # full run
python experiments/make_revision_tables.py                   # tables + figure
python experiments/analyze_revision.py                       # paired tests, detectability
python experiments/analyze_novelty.py                        # group NDCG, gate, cost forecast, Fig1/Fig2
python experiments/run_ablation.py --dataset ml-10m          # scaling-strategy ablation (K=20)
python experiments/export_artifacts.py --dataset all         # exact splits, groups, catalogs, shares
python experiments/run_revision.py --dataset all --save_factors   # also store trained MF factors
```

Protocol (`src/revision.py`):

- **Split.** Temporal 3-way split per user: test = 5 most recent interactions,
  validation = the 5 before them, training = the rest. Split seed 0; MF seeds 0-4.
- **No leakage.** The base MF is trained on training data only, harsh/generous
  groups are defined on training data only, (gamma_H, gamma_G) are tuned on the
  validation fold, and the test fold is evaluated once with frozen factors.
- **Candidate set.** Items with >= 100 training interactions (all datasets).
  Relevant held-out items outside this catalog are excluded from the Recall
  denominator; users with no relevant in-catalog item are not evaluated.
  Books additionally keeps users with >= 100 reviews in `Books_5.json` before
  splitting. ML-10M uses the first 10,000 distinct user IDs in file order.
- **Already-observed items** (training items when ranking for validation;
  training + validation items when ranking for test) are excluded from the full
  ranking. `--no_mask_seen` restores the original behaviour (no exclusion).
- **Tuning.** Same anti-symmetric sign update as `src/isc.py`
  (step = gamma_lr * (1 + lam_mult) = 0.0199; tolerance
  eps = max(0.2 * |baseline validation gap|, 1e-4); at most 500 iterations).
  A run fails if it does not converge or any gamma leaves [1/5, 5]; a failed
  run reverts to gamma = 1 ("fallback"). Both raw and fallback results are reported.
- **Uncertainty.** User-level bootstrap (1,000 resamples within each group) for
  the gap; t-based 95% CIs across the five MF seeds.
- **Comparators.** Validation grid search over (gamma_H, gamma_G) in
  [1/4, 4]^2 (17 x 17 log grid) at K=20; an L2-regularised MF whose
  (L2, d) is chosen on validation RMSE from {0.001, 0.01, 0.1} x {5, 32}.
- **Diagnostics.** Item-bias-only and popularity rankers, group descriptives,
  and gamma_acc = B_g / A_g on train / validation / test.
- **Validation gate.** Calibrate only when the user-level bootstrap interval of
  the validation gap excludes zero (`val_measurable` in the CSV); otherwise keep gamma = 1.
- **Cost forecast.** dMSE_g = A_g (g-1)^2 + 2 A_g (1 - gamma_acc_g)(g-1), evaluated
  with validation A_g and gamma_acc_g, weighted by the group shares of held-out pairs.
- **Ablation.** ISC versus rank-equivalent item-bias scaling, whole-score scaling,
  user-bias shift, a single global factor, and one-sided factors.

## Method summary

For user `u`, item `i`, biased MF predicts `r = mu + b_u + b_i + p_u . q_i`.
ISC replaces the interaction term with `gamma_g(u) * (p_u . q_i)` and tunes
`(gamma_H, gamma_G)` by:

```
a = gamma_lr * (1 + lambda / lambda_max),   lambda = lam_mult * lambda_max
repeat:  s = sign(RecallGap_H - RecallGap_G)
         gamma_H *= (1 + a*s);  gamma_G /= (1 + a*s)
until |gap| <= eps = max(0.2 * |gap_baseline|, 1e-4)
```

with the stability bound `lambda_max = A_G / C_G` computed from the interaction
scores. See `src/isc.py`.

## License

MIT — see `LICENSE`.
