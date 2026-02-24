
from typing import Tuple, Optional

import os
import os.path as osp

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
 
def save_vis(
    point_cloud: torch.Tensor,
    rotated_pcd: torch.Tensor,
    save_dir: str = "vis_pcd",
):
    import os
    import open3d as o3d
    """
    Save original + rotated point clouds as .pcd files.

    point_cloud : (M,3)
    rotated_pcd : (B,M,3)
    """

    os.makedirs(save_dir, exist_ok=True)

    pc = point_cloud.detach().cpu().numpy()
    B = rotated_pcd.shape[0]

    # original cloud
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(pc)
    pcd_ref.paint_uniform_color([0.2, 0.7, 1.0])  # blue

    for i in range(B):
        rot = rotated_pcd[i].detach().cpu().numpy()

        pcd_rot = o3d.geometry.PointCloud()
        pcd_rot.points = o3d.utility.Vector3dVector(rot)
        pcd_rot.paint_uniform_color([1.0, 0.3, 0.3])  # red

        combined = o3d.geometry.PointCloud()
        combined += pcd_ref
        combined += pcd_rot

        filename = os.path.join(save_dir, f"rotation_{i:04d}.pcd")
        o3d.io.write_point_cloud(filename, combined)

    print(f"Saved {B} visualization files to {save_dir}")

def resolve_mesh_root(mesh_root_arg: Optional[str], script_dir: str, repo_root: str) -> Optional[str]:
    """환경변수·기본 경로 후보를 순서대로 확인해 존재하는 디렉터리 반환."""
    candidates = []
    if mesh_root_arg and mesh_root_arg.strip():
        candidates.append(osp.abspath(mesh_root_arg.strip()))
    if os.environ.get('SHAPENET_ROOT'):
        candidates.append(osp.abspath(os.environ['SHAPENET_ROOT']))
    if os.environ.get('MESH_ROOT'):
        candidates.append(osp.abspath(os.environ['MESH_ROOT']))
    candidates.extend([
        osp.join(repo_root, 'ShapeNetCore.v2'),
        osp.join(repo_root, '..', 'ShapeNetCore.v2'),
        osp.join(script_dir, 'ShapeNetCore.v2'),
        osp.abspath(osp.join(os.getcwd(), 'ShapeNetCore.v2')),
    ])
    seen = set()
    for p in candidates:
        p = osp.abspath(p)
        if p in seen:
            continue
        seen.add(p)
        if osp.isdir(p):
            return p
    return None