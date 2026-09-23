"""Shared preparation flow used by every experiment driver.

Order (kept identical to the reference pipeline):
    load split  ->  group users on RAW train  ->  filter items
    ->  build index on filtered train  ->  train biased MF.
Groups are defined before item filtering; training and evaluation use the
item-filtered frame.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from . import data as D
from . import model as M


@dataclass
class Prepared:
    train: pd.DataFrame           # item-filtered training frame
    train_raw: pd.DataFrame       # pre-filter training frame (grouping basis)
    val: pd.DataFrame             # validation fold (empty unless three_way)
    test: pd.DataFrame            # test fold
    groups: dict
    users: list
    items: list
    u2i: dict
    i2i: dict
    mf: object
    device: object


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def prepare(cfg, seed=None, three_way=False, device=None, train=True, progress=True):
    seed = cfg.seed if seed is None else seed
    device = M.get_device() if device is None else device
    set_seed(seed)

    train_raw, val, test = D.load_split(cfg, seed=seed, three_way=three_way)

    # groups on the raw frame; for the 3-way protocol they use train+val
    group_basis = pd.concat([train_raw, val]) if three_way else train_raw
    groups = D.group_users(group_basis, cfg.rel_low, cfg.rel_high)

    train_df = D.filter_items(train_raw, cfg.min_item_ratings)
    users, items, u2i, i2i = M.build_index(train_df)

    mf = None
    if train:
        mf = M.train_mf(train_df, users, items, u2i, i2i, cfg, device, progress=progress)

    return Prepared(train=train_df, train_raw=train_raw, val=val, test=test,
                    groups=groups, users=users, items=items, u2i=u2i, i2i=i2i,
                    mf=mf, device=device)


def group_sizes(groups):
    from collections import Counter
    c = Counter(groups.values())
    return {"harsh": c.get("harsh", 0), "generous": c.get("generous", 0),
            "neutral": c.get("neutral", 0)}
