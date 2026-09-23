"""Dataset loading, temporal splitting, and behavioural user grouping.

All loaders return DataFrames with a uniform schema:
    columns = [user_id, item_id, rating, timestamp, rank_order]
where rank_order = 1 is the most recent interaction of that user.

The MovieLens loaders originally named the item column `movie_id`; here every
dataset uses `item_id` so the rest of the code is dataset-agnostic.
"""
import os
import json
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _rank_and_split(df: pd.DataFrame, num_test_items: int, num_val_items: int = 0):
    """Shuffle-stable temporal split by per-user recency rank.

    rank_order == 1 is the newest item. Returns (train, val, test); `val` is
    empty when num_val_items == 0.
    """
    df = df.copy()
    df["rank_order"] = df.groupby("user_id")["timestamp"].rank(
        ascending=False, method="first"
    )
    test = df[df.rank_order <= num_test_items].copy()
    if num_val_items > 0:
        val = df[(df.rank_order > num_test_items)
                 & (df.rank_order <= num_test_items + num_val_items)].copy()
        train = df[df.rank_order > num_test_items + num_val_items].copy()
    else:
        val = df.iloc[0:0].copy()
        train = df[df.rank_order > num_test_items].copy()
    return train, val, test


def load_movielens(path, n_users, seed, num_test_items=5, num_val_items=0):
    """MovieLens 1M / 10M `ratings.dat` (fields separated by '::')."""
    df = pd.read_csv(
        os.path.join(path, "ratings.dat"),
        sep="::", engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    # keep the first `n_users` distinct users (fixed prefix, as in the paper)
    df = df[df.user_id <= df.user_id.unique()[n_users - 1]].copy()
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return _rank_and_split(df, num_test_items, num_val_items)


def load_books(path, data_file, seed, min_user_ratings=100,
               num_test_items=5, num_val_items=0):
    """Amazon Books reviews (one JSON object per line).

    Applies the user-side k-core filter (>= min_user_ratings) here; the
    item-side filter is applied later, alongside MovieLens, in prepare().
    """
    filepath = os.path.join(path, data_file)
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            u, it, r = rec.get("reviewerID"), rec.get("asin"), rec.get("overall")
            if u and it and r is not None:
                records.append({
                    "user_id": u, "item_id": it,
                    "rating": float(r), "timestamp": int(rec.get("unixReviewTime", 0)),
                })
    df = pd.DataFrame(records)

    counts = df.groupby("user_id").size()
    df = df[df.user_id.isin(counts[counts >= min_user_ratings].index)].copy()
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return _rank_and_split(df, num_test_items, num_val_items)


def load_split(cfg, seed=None, three_way=False):
    """Dispatch on cfg.dataset. Returns (train, val, test)."""
    seed = cfg.seed if seed is None else seed
    n_val = cfg.num_val_items if three_way else 0
    if cfg.dataset in ("ml-10m", "ml-1m"):
        return load_movielens(cfg.data_path(), cfg.num_users(), seed,
                              cfg.num_test_items, n_val)
    if cfg.dataset == "books":
        return load_books(cfg.data_path(), cfg.data_file(), seed,
                          cfg.min_user_ratings, cfg.num_test_items, n_val)
    raise ValueError(f"unknown dataset: {cfg.dataset}")


# --------------------------------------------------------------------------
# Behavioural grouping
# --------------------------------------------------------------------------
def group_users(df, rel_low, rel_high):
    """Assign each user to harsh / generous / neutral by relative positivity.

    rho_u = fraction of a user's ratings that exceed the item's mean rating.
    harsh if rho_u < rel_low, generous if rho_u > rel_high, else neutral.
    Must be computed on the RAW training frame (before the item filter) to
    match the reference pipeline.
    """
    item_avg = df.groupby("item_id")["rating"].mean()
    d = df.copy()
    d["above"] = d.rating > d.item_id.map(item_avg)
    ratio = d.groupby("user_id")["above"].mean()
    g = {}
    for u, r in ratio.items():
        g[u] = "harsh" if r < rel_low else "generous" if r > rel_high else "neutral"
    return g


def filter_items(df, min_item_ratings):
    """Keep only items with at least `min_item_ratings` interactions."""
    return df.groupby("item_id").filter(lambda x: len(x) >= min_item_ratings)
