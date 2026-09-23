"""Biased matrix factorization: r_hat = mu + b_u + b_i + gamma_g * (p_u . q_i).

The optional `scales` vector carries the ISC per-user interaction-term factor
gamma_g(u); with scales=None it reduces to standard biased MF.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class RatingDS(Dataset):
    def __init__(self, df, u2i, i2i):
        self.u = torch.tensor(df.user_id.map(u2i).values, dtype=torch.long)
        self.i = torch.tensor(df.item_id.map(i2i).values, dtype=torch.long)
        self.r = torch.tensor(df.rating.values, dtype=torch.float32)

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]


class BiasedMF(nn.Module):
    def __init__(self, n_users, n_items, k):
        super().__init__()
        self.P = nn.Embedding(n_users, k)
        self.Q = nn.Embedding(n_items, k)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.mu = nn.Parameter(torch.zeros(1))

    def forward(self, u, i, scales=None):
        pu, qi = self.P(u), self.Q(i)
        if scales is not None:
            pu = pu * scales[u].unsqueeze(1)
        return self.mu + self.bu(u).squeeze() + self.bi(i).squeeze() + (pu * qi).sum(1)

    def raw_dot(self, u_idx, i_idx):
        """The bare interaction term s_ui = p_u . q_i (no biases)."""
        return (self.P(u_idx) * self.Q(i_idx)).sum(1)


def build_index(train_df):
    users = sorted(train_df.user_id.unique())
    items = sorted(train_df.item_id.unique())
    u2i = {u: i for i, u in enumerate(users)}
    i2i = {m: i for i, m in enumerate(items)}
    return users, items, u2i, i2i


def train_mf(train_df, users, items, u2i, i2i, cfg, device, progress=True):
    """Train biased MF with plain MSE (no regularization, as in the paper)."""
    from tqdm import trange
    mf = BiasedMF(len(users), len(items), cfg.latent_factors).to(device)
    opt = torch.optim.Adam(mf.parameters(), lr=cfg.lr_mf)
    loader = DataLoader(RatingDS(train_df, u2i, i2i),
                        batch_size=cfg.batch_size, shuffle=True)
    it = trange(cfg.epochs_mf, desc="MF", leave=False) if progress else range(cfg.epochs_mf)
    for _ in it:
        for u, i, r in loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            loss = F.mse_loss(mf(u, i), r)
            opt.zero_grad()
            loss.backward()
            opt.step()
    mf.eval()
    return mf
