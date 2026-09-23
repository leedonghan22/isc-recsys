"""Interaction-Score Calibration (ISC) -- Algorithm 1 of the paper.

Rescales only the interaction term by a per-group scalar gamma_g, tuned offline
by an anti-symmetric sign update until the harsh-generous Recall@K gap falls
within tolerance. Requires no retraining and no demographic data.
"""
import numpy as np
import torch

from .metrics import evaluate


@torch.no_grad()
def compute_sK_raw(model, users, items, u2i, K):
    """Quantile of the interaction scores at the top-K cutoff (used by A_g/C_g)."""
    dev = next(model.parameters()).device
    it = torch.arange(len(items), device=dev)
    dots = [model.raw_dot(torch.full((len(items),), u2i[u], dtype=torch.long, device=dev),
                          it).cpu() for u in users]
    alpha = K / len(items)
    return torch.tensor(np.quantile(torch.cat(dots).numpy(), 1 - alpha),
                        dtype=torch.float32, device=dev)


@torch.no_grad()
def compute_ag_cg(model, df, groups, u2i, i2i, sK_raw, beta):
    """Curvature coefficient A_g = E_g[s^2] and the sigmoid-surrogate term C_g
    used to form the stability bound lambda_max = A_G / C_G.
    """
    dev = next(model.parameters()).device
    out = {}
    for g in ("harsh", "generous"):
        sub = df[df.user_id.map(groups) == g]
        if sub.empty:
            out[g] = (0.0, 0.0)
            continue
        u = torch.tensor(sub.user_id.map(u2i).values, dtype=torch.long, device=dev)
        i = torch.tensor(sub.item_id.map(i2i).values, dtype=torch.long, device=dev)
        s = model.raw_dot(u, i)
        a = torch.mean(s ** 2).item()
        sig = torch.sigmoid(beta * (s - sK_raw))
        C = (beta ** 2 / 2) * torch.mean(sig * (1 - sig) * s ** 2).item()
        out[g] = (a, C)
    return out


def tune(model, users, items, u2i, i2i, groups, val_df, cfg, K,
         eval_df=None, record_history=False):
    """Tune (gamma_H, gamma_G) so that |Recall@K gap| <= eps on val_df.

    Returns a result dict with the tuned factors, iteration count, a converged
    flag, and baseline/final metrics (measured on eval_df, defaulting to val_df).
    The sign update is:  gamma_H *= (1+a*s),  gamma_G /= (1+a*s),  s = sign(gap),
    with a = gamma_lr * (1 + lambda/lambda_max) and lambda = lam_mult*lambda_max.
    """
    dev = next(model.parameters()).device
    eval_df = val_df if eval_df is None else eval_df

    sK_raw = compute_sK_raw(model, users, items, u2i, K)
    coef = compute_ag_cg(model, val_df, groups, u2i, i2i, sK_raw, cfg.beta_sig)
    a_G, C_G = coef["generous"]
    lambda_max = a_G / C_G if C_G > 0 else 1.0
    lam = lambda_max * cfg.lam_mult
    step_scale = 1 + lam / lambda_max if lambda_max != 0 else 1.0

    mask_h = torch.tensor([groups.get(u) == "harsh" for u in users], device=dev)
    mask_g = torch.tensor([groups.get(u) == "generous" for u in users], device=dev)

    def metrics_on(df, scales):
        return evaluate(model, users, u2i, items, i2i, groups, scales, df, K,
                        cfg.relevance_threshold)

    ones = torch.ones(len(users), device=dev)
    base = metrics_on(val_df, ones)
    eps = max(abs(base["recall_gap"]) * cfg.epsilon_ratio, 1e-4)

    gH = torch.tensor(1.0, device=dev)
    gG = torch.tensor(1.0, device=dev)
    history = []
    converged, diverged, iters = False, False, 0

    for t in range(1, cfg.max_iter + 1):
        iters = t
        scales = torch.ones(len(users), device=dev)
        scales[mask_h] = gH
        scales[mask_g] = gG
        m = metrics_on(val_df, scales)
        gap = m["recall_gap"]
        if record_history:
            history.append({"iter": t, "gamma_H": gH.item(), "gamma_G": gG.item(),
                            "gap": gap, "rmse": m["rmse"]})
        if abs(gap) <= eps:
            converged = True
            break
        step = cfg.gamma_lr * step_scale * (1.0 if gap > 0 else -1.0)
        gH = gH * (1 + step)
        gG = gG / (1 + step)
        if gH.item() > cfg.gamma_cap:
            diverged = True
            break

    scales = torch.ones(len(users), device=dev)
    scales[mask_h] = gH
    scales[mask_g] = gG
    final = metrics_on(eval_df, scales)
    base_eval = base if eval_df is val_df else metrics_on(eval_df, ones)

    return {
        "gamma_H": gH.item(), "gamma_G": gG.item(),
        "iters": iters, "converged": converged, "diverged": diverged,
        "eps": eps, "lambda_max": lambda_max,
        "baseline": base_eval, "final": final,
        "history": history,
    }
