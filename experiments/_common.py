"""Small helpers shared by the experiment drivers (arg parsing, output paths)."""
import os
import sys
import argparse

# make `src` importable when running `python experiments/run_x.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")

ALL_DATASETS = ["ml-10m", "ml-1m", "books"]


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", choices=ALL_DATASETS + ["all"], default="ml-10m")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=RESULTS_DIR)
    return p


def datasets_from_args(args):
    return ALL_DATASETS if args.dataset == "all" else [args.dataset]


def make_config(dataset, args):
    cfg = Config(dataset=dataset, seed=args.seed)
    for k in ("lam_mult", "gamma_lr", "max_iter", "epsilon_ratio",
              "latent_factors", "epochs_mf", "gamma_cap"):
        if hasattr(args, k) and getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))
    return cfg


def save_csv(df, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return path
