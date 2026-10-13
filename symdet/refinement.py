"""Step (3b) Refinement: re-estimate a hypothesis from all of its inliers.

reflection: normal = principal eigenvector of the covariance of (x' - x) over inliers, d from the mean midpoint;
            repeated until the inlier set stops changing (or `iters`).
rotation:   n = smallest eigenvector of C = sum d d^T, p from (C + lam I) p = sum d (d^T m), theta = circular mean
            (Eq. rot_refine_compact).

"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .generate_hypothesis import reflection_residual, rodrigues, translation


# ======================================================================
# Reflection
# ======================================================================
def reflection_pca_plane(inl: torch.Tensor, x64: torch.Tensor, xp64: torch.Tensor, c64: torch.Tensor):
    """Plane from an inlier mask: n = top eigenvector of cov(x - x'); d_val = -n . mean(c)  (plane n.x + d_val = 0).
    eigh runs in numpy (same LAPACK path as the original -> identical eigenvector signs)."""
    diffs = x64[inl] - xp64[inl]
    dc = diffs - diffs.mean(dim=0)
    w, v = np.linalg.eigh((dc.T @ dc).cpu().numpy())
    pdir = v[:, np.argmax(w)]
    n_vec = torch.from_numpy(pdir / (np.linalg.norm(pdir) + 1e-8)).to(x64.device, dtype=torch.float64)
    mean_mid = c64[inl].mean(dim=0)
    return n_vec, mean_mid, -(n_vec @ mean_mid)


def refine_reflection(n_vec, mean_mid, d_val, inlier: torch.Tensor, pool: torch.Tensor,
                      x64: torch.Tensor, xp64: torch.Tensor, c64: torch.Tensor, thr: float, iters: int):
    """Iterate PCA re-fit -> re-collect inliers (within `pool`) until stable or `iters` reached."""
    for _ in range(int(iters)):
        if int(inlier.sum().item()) < 3:
            break
        n_vec, mean_mid, d_val = reflection_pca_plane(inlier, x64, xp64, c64)
        refined = (reflection_residual(n_vec, d_val, x64, xp64) < thr) & pool
        if torch.equal(refined, inlier):
            break
        inlier = refined
    return n_vec, mean_mid, d_val, inlier


# ======================================================================
# Rotation
# ======================================================================
@torch.no_grad()
def refine_rotation(P: torch.Tensor, Q: torch.Tensor, n: torch.Tensor, p: torch.Tensor, theta: torch.Tensor,
                    thr: float, iters: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched refinement of (n, p, theta) on pairs (P, Q) with hard inlier weights. Returns n, p, theta, inlier_count."""
    dtype, B, N = P.dtype, n.shape[0], P.shape[0]
    I3 = torch.eye(3, device=P.device, dtype=dtype).unsqueeze(0).expand(B, 3, 3)
    d = P - Q
    m = 0.5 * (P + Q)
    dm = (d * m).sum(dim=1)
    thr_t = torch.tensor(thr, device=P.device, dtype=dtype)

    def _resid(n_, p_, th_):
        R = rodrigues(n_, th_)
        pred = (R @ P.t().unsqueeze(0).expand(B, 3, N)).transpose(1, 2) + translation(R, p_).unsqueeze(1)
        return torch.linalg.norm(pred - Q.unsqueeze(0), dim=2)

    for _ in range(iters):
        w = (_resid(n, p, theta) < thr_t).to(dtype)
        C = torch.einsum("bn,ni,nj->bij", w, d, d)
        rhs = torch.einsum("bn,ni,n->bi", w, d, dm)
        _, evecs = torch.linalg.eigh(C + 1e-6 * I3)
        n = evecs[:, :, 0]
        n = n / (torch.linalg.norm(n, dim=1, keepdim=True) + 1e-12)
        p = torch.linalg.solve(C + 1e-6 * I3, rhs.unsqueeze(-1)).squeeze(-1)
        p = p - (p * n).sum(dim=1, keepdim=True) * n

        v = P.unsqueeze(0) - p.unsqueeze(1)
        u = Q.unsqueeze(0) - p.unsqueeze(1)
        v_perp = v - (v * n.unsqueeze(1)).sum(dim=2, keepdim=True) * n.unsqueeze(1)
        u_perp = u - (u * n.unsqueeze(1)).sum(dim=2, keepdim=True) * n.unsqueeze(1)
        denom = (torch.linalg.norm(v_perp, dim=2).clamp_min(1e-9) * torch.linalg.norm(u_perp, dim=2).clamp_min(1e-9)).clamp_min(1e-9)
        c = (v_perp * u_perp).sum(dim=2) / denom
        s = (torch.cross(v_perp, u_perp, dim=2) * n.unsqueeze(1)).sum(dim=2) / denom
        theta = torch.atan2((w * s).sum(dim=1), (w * c).sum(dim=1))

    return n, p, theta, (_resid(n, p, theta) < thr_t).sum(dim=1)
