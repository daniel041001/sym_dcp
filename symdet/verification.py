"""Step (3a) Verification: symmetry distance error (SDE) and inlier-ratio gates.

SDE(X, T_h) = bidirectional squared Chamfer distance between X and T_h(X)   (Eq. sde_method)
  - pool pruning (tau^0): raw hypotheses, before the loop
  - gate (tau):           after the 1st refinement
Inlier ratio r(h) = |I| / |C_sym| >= rho_min, plus the minimum-support stop.
"""
from __future__ import annotations

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points


# ======================================================================
# Reflection
# ======================================================================
@torch.no_grad()
def reflection_sde_batch(X: torch.Tensor, n: torch.Tensor, d: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """SDE for many planes n_i . x = d_i (pool pruning). X:(M,3) n:(N,3) d:(N,) -> (N,)."""
    N = n.shape[0]
    out = torch.full((N,), float("inf"), dtype=torch.float32, device=n.device)
    I3 = torch.eye(3, device=n.device, dtype=torch.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        n_c, d_c = n[s:e], d[s:e]
        Xs = X.unsqueeze(0) - (d_c.unsqueeze(1) * n_c).unsqueeze(1)
        H = I3.unsqueeze(0) - 2.0 * torch.bmm(n_c.unsqueeze(2), n_c.unsqueeze(1))
        ch, _ = chamfer_distance(Xs, torch.bmm(Xs, H), batch_reduction=None)
        out[s:e] = ch
    return out


@torch.no_grad()
def reflection_sde(X: torch.Tensor, n32: torch.Tensor, shift: torch.Tensor) -> float:
    """SDE of one plane after translating X by `shift` so the plane passes through the origin."""
    Xs = X + shift.unsqueeze(0)
    H = torch.eye(3, device=X.device, dtype=X.dtype) - 2.0 * torch.outer(n32, n32)
    ch, _ = chamfer_distance(Xs.unsqueeze(0), (Xs @ H).unsqueeze(0))
    return float(ch.item())


def reflection_ratio_ok(n_inliers: int, n_pool: int, ratio_pct: float) -> bool:
    """Refined inliers >= ratio_pct % of the remaining pool."""
    return n_inliers >= float(n_pool) * ratio_pct / 100.0


# ======================================================================
# Rotation
# ======================================================================
@torch.no_grad()
def rotation_sde(X: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """SDE for a batch of rigid motions. X:(M,3) R:(B,3,3) t:(B,3) -> (B,)."""
    B, M = R.shape[0], X.shape[0]
    Xr = (R @ X.t().contiguous().unsqueeze(0).expand(B, 3, M)).transpose(1, 2) + t.unsqueeze(1)
    Xb = X.unsqueeze(0).expand(B, -1, -1)
    dx = knn_points(Xb, Xr, K=1).dists[..., 0].mean(dim=1)
    dy = knn_points(Xr, Xb, K=1).dists[..., 0].mean(dim=1)
    return dx + dy


def rotation_gate(theta: torch.Tensor, n_inliers: torch.Tensor, sde: torch.Tensor, n_pool: int,
                  identity_theta: float, sde_thr: float, ratio_pct: float) -> torch.Tensor:
    """After the 1st refinement: non-identity, SDE < tau, inlier ratio >= rho_min."""
    ok = (theta.abs() > identity_theta) & (sde < sde_thr)
    return ok & (n_inliers.float() / float(n_pool) >= float(ratio_pct) / 100.0)


def min_support(n_pool: int, pct: float, floor: int) -> int:
    """Rotation candidates need >= max(floor, ceil(pct % of the remaining pool)) supporters."""
    return max(floor, int(np.ceil(n_pool * float(pct) / 100.0)))
