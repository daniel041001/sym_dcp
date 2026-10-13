"""Step (4) Selection: accept a symmetry and remove its supporting correspondences from the pool.

reflection: the first candidate that passes all gates is accepted; its inlier pairs are removed (rows and columns).
rotation:   among the candidates that survive the gates in one step, take the min-SDE one; then remove pairs
            whose distance-to-axis and height-along-axis agree within tau:
                sqrt((r(x) - r(x'))^2 + (h(x) - h(x'))^2) < tau
"""
from __future__ import annotations

import torch


# ======================================================================
# Reflection
# ======================================================================
def remove_reflection_support(inlier: torch.Tensor, avail: torch.Tensor, valid: torch.Tensor, inlier_matrix: torch.Tensor) -> None:
    """In place: drop accepted inliers from the pool, as candidates (rows) and as supporters (columns)."""
    idx = torch.nonzero(inlier, as_tuple=False).squeeze(1)
    avail[idx] = False
    inlier_matrix[:, idx] = 0.0
    inlier_matrix[idx, :] = 0.0
    valid[idx] = False


# ======================================================================
# Rotation
# ======================================================================
def select_rotation(sde: torch.Tensor) -> int:
    """Among the candidates that survived this step, take the one with the smallest SDE."""
    return int(torch.argmin(sde).item())


@torch.no_grad()
def rotation_removal_mask(P: torch.Tensor, Q: torch.Tensor, n: torch.Tensor, p: torch.Tensor,
                          avail: torch.Tensor, tau: float) -> torch.Tensor:
    """Pairs whose distance-to-axis and height-along-axis agree: sqrt((r(x)-r(x'))^2 + (h(x)-h(x'))^2) < tau."""
    v, u = P - p, Q - p
    vn, un = v @ n, u @ n                                    # heights along the axis
    r_v = torch.linalg.norm(v - vn.unsqueeze(1) * n, dim=1)  # distances to the axis
    r_u = torch.linalg.norm(u - un.unsqueeze(1) * n, dim=1)
    return (torch.sqrt((r_v - r_u) ** 2 + (vn - un) ** 2) < float(tau)) & avail
