
from typing import Tuple

import torch


# -----------------------------
# Correct Rodrigues (SO(3))
# -----------------------------
def rodrigues_torch(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues rotation formula.
    axis: (3,) unit vector (will be normalized internally)
    angle: scalar tensor
    return: (3, 3) rotation matrix in SO(3)
    """
    axis = axis.to(dtype=torch.float32)
    axis = axis / (torch.linalg.norm(axis) + 1e-8)
    ax, ay, az = axis[0], axis[1], axis[2]
 
    zero = torch.tensor(0.0, device=axis.device, dtype=axis.dtype)
    K = torch.stack([
        torch.stack([zero, -az,  ay]),
        torch.stack([az,  zero, -ax]),
        torch.stack([-ay, ax,  zero]),
    ])
 
    I = torch.eye(3, device=axis.device, dtype=axis.dtype)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
 
    # R = I + sinθ K + (1 - cosθ) K^2
    R = I + sin_a * K + (1.0 - cos_a) * (K @ K)
    return R
 
 
# ----------------------------------------
# GPU batched Kabsch / axis-angle utilities
# ----------------------------------------
@torch.no_grad()
def _kabsch_batched(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Batched Kabsch (SVD) on GPU.
    P, Q: (B, K, 3)
    return: R (B, 3, 3)
    """
    P_c = P - P.mean(dim=1, keepdim=True)
    Q_c = Q - Q.mean(dim=1, keepdim=True)
    H = P_c.transpose(1, 2) @ Q_c  # (B, 3, 3)
 
    U, S, Vh = torch.linalg.svd(H)  # batch SVD
    V = Vh.transpose(1, 2)
    R = V @ U.transpose(1, 2)
 
    det = torch.linalg.det(R)  # (B,)
    neg = det < 0
    if neg.any():
        V_fix = V.clone()
        V_fix[neg, :, 2] *= -1.0
        R = V_fix @ U.transpose(1, 2)
    return R
 
 
@torch.no_grad()
def _axis_angle_from_R_batched(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    R: (B, 3, 3)
    returns: axis (B, 3), angle (B,)
    """
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos = (tr - 1.0) * 0.5
    cos = torch.clamp(cos, -1.0, 1.0)
    angle = torch.arccos(cos)
 
    sin = torch.sin(angle)
    sin_safe = torch.where(sin.abs() < 1e-6, torch.ones_like(sin), sin)
    K = (R - R.transpose(1, 2)) / (2.0 * sin_safe).view(-1, 1, 1)
 
    axis = torch.stack([K[:, 2, 1], K[:, 0, 2], K[:, 1, 0]], dim=1)
    axis = axis / (torch.linalg.norm(axis, dim=1, keepdim=True) + 1e-8)
    return axis, angle
 
