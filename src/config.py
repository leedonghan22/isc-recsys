"""Central configuration: dataset locations and shared hyper-parameters.

Dataset paths are read from configs/datasets.yaml (edit that file once to point
at your local data). Everything else defaults to the values used in the paper
and can be overridden per-run from the command line.
"""
import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASETS_YAML = os.path.join(_ROOT, "configs", "datasets.yaml")


def load_dataset_paths() -> dict:
    if not os.path.exists(_DATASETS_YAML):
        raise FileNotFoundError(
            f"{_DATASETS_YAML} not found. Copy configs/datasets.example.yaml to "
            f"configs/datasets.yaml and set your local dataset paths."
        )
    with open(_DATASETS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    # dataset selector: one of {"ml-10m", "ml-1m", "books"}
    dataset: str = "ml-10m"

    # data splitting
    num_test_items: int = 5          # temporal leave-k-out: last k items -> test
    num_val_items: int = 5           # for the 3-way (val->test) protocol
    min_item_ratings: int = 100      # keep items with >= this many ratings
    min_user_ratings: int = 100      # (Books) keep users with >= this many ratings
    relevance_threshold: int = 3     # rating >= this is "relevant"

    # groups
    rel_low: float = 0.2             # harsh if positivity rho_u < rel_low
    rel_high: float = 0.8            # generous if rho_u > rel_high

    # matrix factorization
    latent_factors: int = 5
    epochs_mf: int = 50
    lr_mf: float = 0.01
    batch_size: int = 512

    # ISC
    beta_sig: float = 20.0           # sigmoid bandwidth used for A_g / C_g
    gamma_lr: float = 0.01           # base step eta
    lam_mult: float = 0.99           # lambda = lam_mult * lambda_max
    max_iter: int = 500
    epsilon_ratio: float = 0.2       # eps = max(epsilon_ratio*|gap_BL|, 1e-4)
    gamma_cap: float = float("inf")  # optional hard cap on gamma_H (runaway guard)

    # evaluation
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    anchor_k: int = 20

    seed: int = 0

    def num_users(self) -> Optional[int]:
        """Per-dataset user cap (MovieLens uses a fixed user prefix; Books uses all)."""
        return {"ml-10m": 10_000, "ml-1m": 6_040, "books": None}[self.dataset]

    def data_path(self) -> str:
        return load_dataset_paths()[self.dataset]["path"]

    def data_file(self) -> Optional[str]:
        return load_dataset_paths()[self.dataset].get("file")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["num_users"] = self.num_users()
        return d
