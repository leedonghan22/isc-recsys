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
