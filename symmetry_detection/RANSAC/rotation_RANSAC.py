from typing import Tuple

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance

try:
    from ..utils.utils import _axis_angle_from_R_batched, _kabsch_batched
except ImportError:
    from symmetry_detection.utils.utils import _axis_angle_from_R_batched, _kabsch_batched


@torch.no_grad()
def _rotations_from_pairs_gpu(
    base_corr_np: np.ndarray,
    rotated_corr_np: np.ndarray,
    point_cloud: torch.Tensor,          # (M,3), centered, on GPU
    device: torch.device,
    inlier_dist_threshold: float = 0.05,
    num_iterations: int = 50000,
    seed_pairs: int = 3,
    batch_hyp: int = 256,
    topk_chamfer: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU batched RANSAC-like rotation extraction from correspondence pairs.
 
    - 모든 inlier 계산은 GPU 벡터화
    - chamfer는 inlier count 상위 top-k 가설만 계산(동기화 비용 최소화)
    - NOTE: 기존 CPU 버전처럼 "inlier 제거"는 GPU 성능 저하가 커서 하지 않음.
            대신 많은 후보를 뽑고, chamfer + threshold로 필터링 후 evaluate 단계에서 사용.
    """
    n = int(base_corr_np.shape[0])
    if n < seed_pairs:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
 
    # Move correspondences to GPU once
    P_all = torch.from_numpy(base_corr_np).to(device=device, dtype=torch.float32)      # (N,3)
    Q_all = torch.from_numpy(rotated_corr_np).to(device=device, dtype=torch.float32)  # (N,3)
 
    thr = torch.tensor(inlier_dist_threshold, device=device, dtype=torch.float32)
    num_batches = (num_iterations + batch_hyp - 1) // batch_hyp
 
    axes_list, angles_list, chamfers_list = [], [], []
 
    # Precompute transposes for faster matmul
    P_all_t = P_all.t().contiguous()  # (3,N)
    Q_all_t = Q_all.t().contiguous()  # (3,N)
 
    for _ in range(num_batches):
        # sample seeds: (B, K)
        idx = torch.randint(0, n, (batch_hyp, seed_pairs), device=device)
 
        P_seed = P_all[idx]  # (B,K,3)
        Q_seed = Q_all[idx]  # (B,K,3)
 
        # Kabsch for all hypotheses
        R = _kabsch_batched(P_seed, Q_seed)  # (B,3,3)
 
        # RP: (B,N,3), RTQ: (B,N,3)
        RP = (R @ P_all_t).transpose(1, 2)                       # (B,N,3)
        RTQ = (R.transpose(1, 2) @ Q_all_t).transpose(1, 2)      # (B,N,3)
 
        d1 = torch.linalg.norm(RP - Q_all.unsqueeze(0), dim=2)   # (B,N)
        d2 = torch.linalg.norm(RTQ - P_all.unsqueeze(0), dim=2)  # (B,N)
        min_d = torch.minimum(d1, d2)
 
        inlier_count = (min_d < thr).sum(dim=1)  # (B,)
 
        k = min(topk_chamfer, batch_hyp)
        top_vals, top_idx = torch.topk(inlier_count, k=k, largest=True)
 
        if top_vals.max().item() < seed_pairs:
            continue
 
        R_top = R[top_idx]  # (k,3,3)
        axis_top, angle_top = _axis_angle_from_R_batched(R_top)
 
        # Rotate point cloud for each hypothesis: (k,M,3)
        pc = point_cloud  # (M,3)
        M = pc.shape[0]
        pc_t = pc.t().unsqueeze(0).expand(k, 3, M)               # (k,3,M)
        pc_rot = (R_top @ pc_t).transpose(1, 2).contiguous()     # (k,M,3)
 
        # Chamfer per hypothesis (k is small)
        chamfers = []
        for t in range(k):
            ch, _ = chamfer_distance(pc.unsqueeze(0), pc_rot[t].unsqueeze(0))
            chamfers.append(ch)
        chamfers = torch.stack(chamfers).view(-1)  # (k,)
 
        axes_list.append(axis_top.detach().cpu())
        angles_list.append(angle_top.detach().cpu())
        chamfers_list.append(chamfers.detach().cpu())
 
    if len(axes_list) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
 
    axes = torch.cat(axes_list, dim=0).numpy().astype(np.float32)
    angles = torch.cat(angles_list, dim=0).numpy().astype(np.float32)
    chamfers = torch.cat(chamfers_list, dim=0).numpy().astype(np.float32)
    return axes, angles, chamfers
