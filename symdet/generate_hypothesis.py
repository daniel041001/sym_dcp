"""Step (2) Hypothesis generation: minimal sample -> symmetry hypothesis, and its inlier set (Eq. inlier_generic).

reflection: one pair (x, x') -> plane (n, d),  n = (x - x') / ||x - x'||,  mid c = (x + x') / 2,  d = n . c
            every correspondence is a hypothesis (exhaustive).
rotation:   two pairs -> axis (n, p) and angle theta  (random minimal samples).

"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch


# ======================================================================
# Reflection
# ======================================================================
def reflection_hypotheses(x: np.ndarray, x_prime: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One plane per correspondence. Returns normals n (N,3), midpoints c (N,3), offsets d = n.c (N,)."""
    diff = x - x_prime
    n = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8)
    c = 0.5 * (x + x_prime)
    d = np.sum(n * c, axis=1)
    return n, c, d


def reflection_residual(n_vec: torch.Tensor, d_val: torch.Tensor, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
    """e((x,x'); h) for plane n.x + d_val = 0: mean of ||H x - x'|| and ||H x' - x||  (float64 in the loop)."""
    rx = x - 2.0 * ((x @ n_vec + d_val).unsqueeze(1) * n_vec.unsqueeze(0))
    rxp = x_prime - 2.0 * ((x_prime @ n_vec + d_val).unsqueeze(1) * n_vec.unsqueeze(0))
    return (torch.linalg.norm(rx - x_prime, dim=1) + torch.linalg.norm(rxp - x, dim=1)) * 0.5


@torch.no_grad()
def reflection_inlier_matrix(n: torch.Tensor, d: torch.Tensor, x: torch.Tensor, x_prime: torch.Tensor,
                             thr: float, chunk: int = 128) -> torch.Tensor:
    """(N_hyp, N_pairs) float32 0/1 matrix: pair j is an inlier of hypothesis i (plane n_i . x = d_i)."""
    N = n.shape[0]
    out = torch.zeros((N, x.shape[0]), dtype=torch.float32, device=n.device)
    I3 = torch.eye(3, device=n.device, dtype=torch.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        n_c, d_c = n[s:e], d[s:e]
        H = I3.unsqueeze(0) - 2.0 * torch.bmm(n_c.unsqueeze(2), n_c.unsqueeze(1))
        shift = d_c.unsqueeze(1) * n_c
        xs = x.unsqueeze(0) - shift.unsqueeze(1)
        xps = x_prime.unsqueeze(0) - shift.unsqueeze(1)
        e1 = torch.norm(torch.bmm(xs, H) - xps, dim=2)
        e2 = torch.norm(torch.bmm(xps, H) - xs, dim=2)
        out[s:e, :] = ((e1 + e2) / 2.0 < thr).to(torch.float32)
    return out


# ======================================================================
# Rotation
# ======================================================================
@torch.no_grad()
def rodrigues(n: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """n:(B,3) unit axis, theta:(B,) -> R:(B,3,3)."""
    B = n.shape[0]
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    K = torch.zeros((B, 3, 3), device=n.device, dtype=n.dtype)
    K[:, 0, 1] = -nz; K[:, 0, 2] = ny
    K[:, 1, 0] = nz;  K[:, 1, 2] = -nx
    K[:, 2, 0] = -ny; K[:, 2, 1] = nx
    I = torch.eye(3, device=n.device, dtype=n.dtype).unsqueeze(0).expand(B, -1, -1)
    return I + torch.sin(theta).view(B, 1, 1) * K + (1.0 - torch.cos(theta).view(B, 1, 1)) * (K @ K)


def translation(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """t = p - R p  (Eq. axis_angle_rt)."""
    return p - (R @ p.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def rotation_from_2pairs(P_seed: torch.Tensor, Q_seed: torch.Tensor, eps_cross: float = 1e-3, eps_norm: float = 1e-3):
    """Two pairs (B,2,3) -> n, p, theta, R, t, valid.
    n ∝ d1 x d2 ; p = min-norm solution of d_i^T p = d_i^T m_i (gauge-fixed) ; theta from pair 0."""
    B = P_seed.shape[0]
    d = P_seed - Q_seed
    d1, d2 = d[:, 0], d[:, 1]
    valid = (torch.linalg.norm(d1, dim=1) > eps_norm) & (torch.linalg.norm(d2, dim=1) > eps_norm)
    n_raw = torch.cross(d1, d2, dim=1)
    n_norm = torch.linalg.norm(n_raw, dim=1)
    valid = valid & (n_norm > eps_cross)
    n = n_raw / (n_norm.view(-1, 1) + 1e-12)

    m = 0.5 * (P_seed + Q_seed)
    b = (d * m).sum(dim=2)
    AA = d @ d.transpose(1, 2) + 1e-6 * torch.eye(2, device=d.device, dtype=d.dtype).unsqueeze(0).expand(B, -1, -1)
    u = torch.linalg.solve(AA, b.unsqueeze(-1)).squeeze(-1)
    p = (d.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)
    p = p - (p * n).sum(dim=1, keepdim=True) * n

    v = P_seed[:, 0] - p
    w = Q_seed[:, 0] - p
    v_perp = v - (v * n).sum(dim=1, keepdim=True) * n
    w_perp = w - (w * n).sum(dim=1, keepdim=True) * n
    nv, nw = torch.linalg.norm(v_perp, dim=1), torch.linalg.norm(w_perp, dim=1)
    valid = valid & (nv > eps_norm) & (nw > eps_norm)
    denom = (nv * nw).clamp_min(1e-12)
    theta = torch.atan2((n * torch.cross(v_perp, w_perp, dim=1)).sum(dim=1) / denom, (v_perp * w_perp).sum(dim=1) / denom)

    R = rodrigues(n, theta)
    return n, p, theta, R, translation(R, p), valid


@torch.no_grad()
def iter_rotation_hypotheses(P: torch.Tensor, Q: torch.Tensor, num: int, batch: int, seed_pairs: int = 2) -> Iterator[Tuple]:
    """Yield batches (n, p, theta, R, t) of valid hypotheses from random minimal samples.
    """
    N = P.shape[0]
    for _ in range((num + batch - 1) // batch):
        idx = torch.randint(0, N, (batch, seed_pairs), device=P.device)
        n, p, theta, R, t, valid = rotation_from_2pairs(P[idx], Q[idx])
        if not valid.any():
            continue
        v = torch.nonzero(valid, as_tuple=False).squeeze(1)
        yield n[v], p[v], theta[v], R[v], t[v]


@torch.no_grad()
def rotation_residual(R: torch.Tensor, t: torch.Tensor, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """e((x,x'); h) = min(||R x + t - x'||, ||R^T (x' - t) - x||).  R:(B,3,3) t:(B,3) -> (B,N)."""
    P_t, Q_t = P.t().contiguous(), Q.t().contiguous()
    fwd = torch.linalg.norm((R @ P_t).transpose(1, 2) + t.unsqueeze(1) - Q.unsqueeze(0), dim=2)
    Rinv = R.transpose(1, 2)
    tinv = -(Rinv @ t.unsqueeze(-1)).squeeze(-1)
    bwd = torch.linalg.norm((Rinv @ Q_t).transpose(1, 2) + tinv.unsqueeze(1) - P.unsqueeze(0), dim=2)
    return torch.minimum(fwd, bwd)
