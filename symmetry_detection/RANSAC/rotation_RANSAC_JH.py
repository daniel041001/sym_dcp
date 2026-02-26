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

    ## adaptive num_iterations based on number of correspondences (optional)
    delta = 0.05 # probability of choosing at least one outlier-free subset
    num_iterations = int(np.ceil((n / seed_pairs) * np.log(n / delta)))
    # print(f"num_iterations : {num_iterations} (adaptive based on n={n})")
    num_batches = (num_iterations + batch_hyp - 1) // batch_hyp
 
    axes_list, angles_list, chamfers_list = [], [], []
 
    # Precompute transposes for faster matmul
    P_all_t = P_all.t().contiguous()  # (3,N)
    Q_all_t = Q_all.t().contiguous()  # (3,N)

    candidate_count_filter = point_cloud.shape[0] // 4  # heuristic: 최소 inlier 수 (전체 포인트의 25% 이상)
    eval_hyp_chunk = min(32, batch_hyp)
    
    # RANSAC iterations for inlier counting
    for _ in range(num_batches):
        # sample seeds: (B, K)
        idx = torch.randint(0, n, (batch_hyp, seed_pairs), device=device)
 
        P_seed = P_all[idx]  # (B,K,3)
        Q_seed = Q_all[idx]  # (B,K,3)
 
        # Kabsch for all hypotheses
        R = _kabsch_batched(P_seed, Q_seed)  # (B,3,3)
 
        rotated_pcd = point_cloud.unsqueeze(0) @ R #.transpose(1, 2)  # (B,M,3)
        for s in range(0, batch_hyp, eval_hyp_chunk):
            e = min(s + eval_hyp_chunk, batch_hyp)
            rotated_chunk = rotated_pcd[s:e]  # (C,M,3)
            inlier_matrix_chunk = torch.cdist(point_cloud, rotated_chunk, p=2) < thr  # (C,M,M)
            inlier_count_chunk = inlier_matrix_chunk.sum(dim=(1, 2))  # (C,)

            best_idx_chunk = torch.where(inlier_count_chunk >= candidate_count_filter)[0]
            if best_idx_chunk.numel() == 0:
                continue

            for j in best_idx_chunk.tolist():
                inlier_matrix_whole = inlier_matrix_chunk[j]  # (M,M)
                inlier_src, inlier_trg = torch.nonzero(inlier_matrix_whole, as_tuple=True)
                if inlier_src.numel() < seed_pairs:
                    continue
                R_refined = _kabsch_batched(
                    point_cloud[inlier_src].unsqueeze(0),
                    point_cloud[inlier_trg].unsqueeze(0),
                )[0]  # (3,3)
                rotated_pcd_refined = point_cloud @ R_refined #.transpose(0, 1)  # (M,3)

                axis_refined, angle_refined = _axis_angle_from_R_batched(R_refined.unsqueeze(0))
                ch, _ = chamfer_distance(point_cloud.unsqueeze(0), rotated_pcd_refined.unsqueeze(0))
                axes_list.append(axis_refined.detach().cpu())
                angles_list.append(angle_refined.detach().cpu())
                chamfers_list.append(ch.unsqueeze(0).detach().cpu())

    # from ..utils.utils import save_vis
    # save_vis(point_cloud, torch.cat(best_rotated_pcd, dim=0), "vis_candidates")
    # breakpoint()
    # print("HI")
        # breakpoint()

        # # RP: (B,N,3), RTQ: (B,N,3)
        # RP = (R @ P_all_t).transpose(1, 2)                       # (B,N,3)
        # RTQ = (R.transpose(1, 2) @ Q_all_t).transpose(1, 2)      # (B,N,3)
 
        # d1 = torch.linalg.norm(RP - Q_all.unsqueeze(0), dim=2)   # (B,N)
        # d2 = torch.linalg.norm(RTQ - P_all.unsqueeze(0), dim=2)  # (B,N)
        # min_d = torch.minimum(d1, d2)
 
        # inlier_count = (min_d < thr).sum(dim=1)  # (B,)
 
        # k = min(topk_chamfer, batch_hyp)
        # top_vals, top_idx = torch.topk(inlier_count, k=k, largest=True)
 
        # if top_vals.max().item() < seed_pairs:
        #     continue
 
        # R_top = R[top_idx]  # (k,3,3)
        # axis_top, angle_top = _axis_angle_from_R_batched(R_top)
 
        # # Rotate point cloud for each hypothesis: (k,M,3)
        # pc = point_cloud  # (M,3)
        # M = pc.shape[0]
        # pc_t = pc.t().unsqueeze(0).expand(k, 3, M)               # (k,3,M)
        # pc_rot = (R_top @ pc_t).transpose(1, 2).contiguous()     # (k,M,3)
 
        # # Chamfer per hypothesis (k is small)
        # chamfers = []
        # for t in range(k):
        #     ch, _ = chamfer_distance(pc.unsqueeze(0), pc_rot[t].unsqueeze(0))
        #     chamfers.append(ch)
        # chamfers = torch.stack(chamfers).view(-1)  # (k,)
 
        # axes_list.append(axis_top.detach().cpu())
        # angles_list.append(angle_top.detach().cpu())
        # chamfers_list.append(chamfers.detach().cpu())
 
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
