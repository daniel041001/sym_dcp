"""
DCP 모델을 사용하여 correspondence를 추출하고,
ShapeNet(SymmetryNet GT)에서 rotation symmetry axis detection을 평가하는 단일 파일 스크립트.

핵심 변경(요청사항):
- RANSAC minimal set: 3pair(Kabsch) -> 2pair로 (n,p,theta) 초기 가설 생성
- Refinement: 일반 registration(Kabsch) 대신, rotation symmetry 제약을 유지한 (n,p,theta) refinement
  * n: smallest-eig of C=Σ d d^T (d=x-y)
  * p: (C+λI)p = Σ d(d^T m), m=(x+y)/2
  * theta: projected vectors로 circular mean
- Inlier counting: forward/backward 중 작은 residual 사용(자기 대응에서 ±theta 혼재에 견고)
- Chamfer: RX+t 포함, 배치 계산(knn_points 기반). 전체 포인트클라우드 사용(서브샘플 없음)

추가 변경(이번 요청):
- batch top-k 선택(topk_hyp) 제거: 배치에서 나온 "유효한 모든 가설"을 refine + chamfer까지 처리
  * 단, 메모리 안정화를 위해 refine_chunk 단위로만 나눠서 처리(결과/기능은 동일, 단지 내부 처리 단위)

실행:
  python dcp_rot_sym_eval_2pair.py --model_path dcp/pretrained/dcp_v2.t7 --device cuda:0 ...

주의:
- repo 구조는 기존 코드와 동일하다고 가정 (repo_root/dcp, repo_root/model 등)
- SymmetryNet GT: dcp/model_symmetry_extracted/model_symmetry/<class>/<instance>/model_sym.txt
- ShapeNetCore.v2 mesh root 필요(캐시 없을 때): --mesh_root 또는 env SHAPENET_ROOT/MESH_ROOT
"""

import os
import os.path as osp
import sys
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch

# ---- repo root 를 path 에 추가 ----
_script_dir = osp.dirname(osp.abspath(__file__))
_repo_root = osp.abspath(osp.join(_script_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# DCP (클래스는 dcp/model.py에 정의됨; dcp에서 실행 시 model은 이 파일)
from model.DCP_base_model import DCP

# SymmetryNet ShapeNet GT 로더 (원본 메시 → 포인트클라우드는 trimesh)
from data.symmetrynet_shapenet_gt import build_shapenet_rot_cache

# pytorch3d (batched chamfer via knn)
from pytorch3d.ops import knn_points

# tqdm (with safe fallback)
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


# ============================================================
#  Batched rotation-symmetry hypothesis (2pair) + refinement
# ============================================================

@torch.no_grad()
def _rodrigues_batched(n: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues formula.
    n: (B,3) unit axis
    theta: (B,)
    return R: (B,3,3)
    """
    B = n.shape[0]
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    K = torch.zeros((B, 3, 3), device=n.device, dtype=n.dtype)
    K[:, 0, 1] = -nz
    K[:, 0, 2] =  ny
    K[:, 1, 0] =  nz
    K[:, 1, 2] = -nx
    K[:, 2, 0] = -ny
    K[:, 2, 1] =  nx

    I = torch.eye(3, device=n.device, dtype=n.dtype).unsqueeze(0).expand(B, -1, -1)
    sin = torch.sin(theta).view(B, 1, 1)
    cos = torch.cos(theta).view(B, 1, 1)
    R = I + sin * K + (1.0 - cos) * (K @ K)
    return R


@torch.no_grad()
def _hypothesize_symmetry_from_2pairs_batched(
    P_seed: torch.Tensor,  # (B,2,3) x
    Q_seed: torch.Tensor,  # (B,2,3) y
    eps_cross: float = 1e-3,
    eps_norm: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    2pair로 (n,p,theta,R,t) 가설 생성.
    - n ∝ (x1-y1)×(x2-y2)
    - p: min-norm solution of d_i^T p = d_i^T m_i
    - theta: pair0로부터 projection 후 atan2
    return:
      n (B,3), p (B,3), theta (B,), R (B,3,3), t (B,3), valid_mask (B,)
    """
    device = P_seed.device
    dtype = P_seed.dtype
    B = P_seed.shape[0]

    d = P_seed - Q_seed                       # (B,2,3), d_i = x_i - y_i
    d1 = d[:, 0]
    d2 = d[:, 1]

    # reject: tiny |d| (x≈y)
    d1n = torch.linalg.norm(d1, dim=1)
    d2n = torch.linalg.norm(d2, dim=1)
    valid = (d1n > eps_norm) & (d2n > eps_norm)

    # n ∝ d1 × d2
    n_raw = torch.cross(d1, d2, dim=1)        # (B,3)
    n_norm = torch.linalg.norm(n_raw, dim=1)
    valid = valid & (n_norm > eps_cross)
    n = n_raw / (n_norm.view(-1, 1) + 1e-12)  # (B,3)

    # Solve for p using min-norm solution:
    # A p = b, A is 2x3 with rows d_i^T, b_i = d_i^T m_i
    m = 0.5 * (P_seed + Q_seed)               # (B,2,3)
    b = (d * m).sum(dim=2)                    # (B,2)

    A = d                                     # (B,2,3)
    AA = A @ A.transpose(1, 2)                # (B,2,2)
    I2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    AA = AA + 1e-6 * I2

    u = torch.linalg.solve(AA, b.unsqueeze(-1)).squeeze(-1)          # (B,2)
    p = (A.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)            # (B,3)

    # gauge fix: remove component along axis direction (choose canonical p)
    p = p - (p * n).sum(dim=1, keepdim=True) * n

    # theta from pair0
    v = P_seed[:, 0] - p    # (B,3)
    w = Q_seed[:, 0] - p    # (B,3)

    v_perp = v - (v * n).sum(dim=1, keepdim=True) * n
    w_perp = w - (w * n).sum(dim=1, keepdim=True) * n

    nv = torch.linalg.norm(v_perp, dim=1)
    nw = torch.linalg.norm(w_perp, dim=1)
    valid = valid & (nv > eps_norm) & (nw > eps_norm)

    denom = (nv * nw).clamp_min(1e-12)
    c = (v_perp * w_perp).sum(dim=1) / denom
    s = (n * torch.cross(v_perp, w_perp, dim=1)).sum(dim=1) / denom
    theta = torch.atan2(s, c)  # (B,)

    R = _rodrigues_batched(n, theta)
    Rp = (R @ p.unsqueeze(-1)).squeeze(-1)
    t = p - Rp

    return n, p, theta, R, t, valid


@torch.no_grad()
def refine_symmetry_batched_with_weights(
    P: torch.Tensor,      # (N,3) all correspondences (x)
    Q: torch.Tensor,      # (N,3) all correspondences (y)
    n: torch.Tensor,      # (B,3) init
    p: torch.Tensor,      # (B,3) init
    theta: torch.Tensor,  # (B,) init
    thr: float,
    iters: int = 3,
    soft_weights: bool = False,
):
    """
    파이썬 루프 없이 배치로 symmetry refinement.
    - weights는 residual 기반으로 (hard inlier) 또는 (soft IRLS) 로 계산
    - n: smallest-eig of C = Σ w d d^T
    - p: (C+λI)p = Σ w d(d^T m)
    - theta: projected vectors circular mean

    반환: (n, p, theta, inlier_count)
    """
    device = P.device
    dtype = P.dtype
    B = n.shape[0]
    N = P.shape[0]
    I3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3)

    d = (P - Q)                 # (N,3)
    m = 0.5 * (P + Q)           # (N,3)
    dm = (d * m).sum(dim=1)     # (N,)

    thr_t = torch.tensor(thr, device=device, dtype=dtype)

    for _ in range(iters):
        # current transform
        R = _rodrigues_batched(n, theta)  # (B,3,3)
        t = p - (R @ p.unsqueeze(-1)).squeeze(-1)  # (B,3)

        # residuals for weights: ||R x + t - y||
        pred = (R @ P.t().unsqueeze(0).expand(B, 3, N)).transpose(1, 2) + t.unsqueeze(1)  # (B,N,3)
        resid = torch.linalg.norm(pred - Q.unsqueeze(0), dim=2)  # (B,N)

        if soft_weights:
            # 부드러운 IRLS
            w = torch.exp(- (resid / thr_t).pow(2))
        else:
            # hard inlier mask
            w = (resid < thr_t).to(dtype)  # (B,N)

        # C[b] = Σ_i w[b,i] * d_i d_i^T  => (B,3,3)
        C = torch.einsum("bn,ni,nj->bij", w, d, d)

        # rhs[b] = Σ_i w[b,i] * d_i * (d_i^T m_i)
        rhs = torch.einsum("bn,ni,n->bi", w, d, dm)

        # n = eigenvector with smallest eigenvalue
        evals, evecs = torch.linalg.eigh(C + 1e-6 * I3)  # (B,3), (B,3,3)
        n = evecs[:, :, 0]
        n = n / (torch.linalg.norm(n, dim=1, keepdim=True) + 1e-12)

        # p solve
        p = torch.linalg.solve(C + 1e-6 * I3, rhs.unsqueeze(-1)).squeeze(-1)
        # gauge fix
        p = p - (p * n).sum(dim=1, keepdim=True) * n

        # theta circular mean (batched)
        v = P.unsqueeze(0) - p.unsqueeze(1)  # (B,N,3)
        u = Q.unsqueeze(0) - p.unsqueeze(1)  # (B,N,3)

        v_perp = v - (v * n.unsqueeze(1)).sum(dim=2, keepdim=True) * n.unsqueeze(1)
        u_perp = u - (u * n.unsqueeze(1)).sum(dim=2, keepdim=True) * n.unsqueeze(1)

        nv = torch.linalg.norm(v_perp, dim=2).clamp_min(1e-9)
        nu = torch.linalg.norm(u_perp, dim=2).clamp_min(1e-9)
        denom = (nv * nu).clamp_min(1e-9)

        c = (v_perp * u_perp).sum(dim=2) / denom
        s = (torch.cross(v_perp, u_perp, dim=2) * n.unsqueeze(1)).sum(dim=2) / denom

        Csum = (w * c).sum(dim=1)
        Ssum = (w * s).sum(dim=1)
        theta = torch.atan2(Ssum, Csum)

    # 최종 inlier count
    R = _rodrigues_batched(n, theta)
    t = p - (R @ p.unsqueeze(-1)).squeeze(-1)
    pred = (R @ P.t().unsqueeze(0).expand(B, 3, N)).transpose(1, 2) + t.unsqueeze(1)
    resid = torch.linalg.norm(pred - Q.unsqueeze(0), dim=2)
    inlier_count = (resid < thr_t).sum(dim=1)  # (B,)

    return n, p, theta, inlier_count


@torch.no_grad()
def _disambiguate_theta_sign(
    P: torch.Tensor, Q: torch.Tensor,
    n: torch.Tensor, p: torch.Tensor, theta: torch.Tensor,
    thr: float
) -> torch.Tensor:
    """
    theta와 -theta 중, forward residual이 더 작은 쪽을 선택.
    """
    thr_t = torch.tensor(thr, device=P.device, dtype=P.dtype)

    def mean_res(th: torch.Tensor) -> torch.Tensor:
        R = _rodrigues_batched(n.view(1, 3), th.view(1))[0]
        t = p - (R @ p)
        pred = (R @ P.t()).t() + t
        res = torch.linalg.norm(pred - Q, dim=1)
        keep = res < (5.0 * thr_t)
        if keep.any():
            return res[keep].mean()
        return res.mean()

    r1 = mean_res(theta)
    r2 = mean_res(-theta)
    return theta if r1 <= r2 else -theta


@torch.no_grad()
def _line_to_line_distance_batched(
    p_pred: torch.Tensor,   # (K,3) one point on predicted axis
    n_pred: torch.Tensor,   # (K,3) unit direction of predicted axis
    n_gt: torch.Tensor,     # (G,3) unit direction of GT axis (line through origin)
) -> torch.Tensor:
    """
    Distance between two 3D lines (batched).
    - GT line: through origin, direction n_gt (unit).
    - Pred line: through p_pred, direction n_pred (unit).
    Returns: (K, G) distances.
    """
    # w = O2 - O1 = p_pred (one point on pred line minus origin)
    # d1 = n_gt (G,3), d2 = n_pred (K,3)
    # Expand: p_pred (K,1,3), n_gt (1,G,3), n_pred (K,1,3)
    K, G = p_pred.shape[0], n_gt.shape[0]
    w = p_pred.unsqueeze(1)                           # (K,1,3)
    d1 = n_gt.unsqueeze(0)                            # (1,G,3)
    d2 = n_pred.unsqueeze(1)                          # (K,1,3)
    c = torch.cross(d1.expand(K, G, 3), d2.expand(K, G, 3), dim=2)  # (K,G,3)
    cn = torch.linalg.norm(c, dim=2).clamp_min(1e-8)   # (K,G)
    # Skew: dist = |w · c| / ||c||
    dist_skew = (w.expand(K, G, 3) * c).sum(dim=2).abs() / cn  # (K,G)
    # Parallel: dist = distance from p_pred to line(origin, n_gt) = ||p_pred × n_gt||
    dist_parallel = torch.linalg.norm(
        torch.cross(p_pred.unsqueeze(1).expand(K, G, 3), d1.expand(K, G, 3), dim=2), dim=2
    )  # (K,G)
    # Use skew when not parallel, else parallel
    parallel = cn < 1e-6
    return torch.where(parallel, dist_parallel, dist_skew)


@torch.no_grad()
def _chamfer_batched_knn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Chamfer distance (squared) per batch using knn_points.
    x, y: (B,N,3)
    return: (B,) mean of squared distances (x->y + y->x)
    """
    x_nn = knn_points(x, y, K=1)
    y_nn = knn_points(y, x, K=1)
    dx = x_nn.dists[..., 0].mean(dim=1)
    dy = y_nn.dists[..., 0].mean(dim=1)
    return dx + dy

@torch.no_grad()
def _rotations_from_pairs_gpu(
    base_corr_np: np.ndarray,
    rotated_corr_np: np.ndarray,
    point_cloud: torch.Tensor,          # (M,3) in SAME coordinates as correspondences
    device: torch.device,
    inlier_dist_threshold: float = 0.007,
    num_iterations: int = 20000,
    seed_pairs: int = 2,
    batch_hyp: int = 2048,
    refine_iters: int = 1,
    chamfer_threshold: float = 0.005,
    identity_theta_thr: float = 1e-2,
    soft_weights: bool = False,
    min_inliers_after_refine: int = 10,
    min_inlier_ratio_after_refine: float = 0.00,
    refine_chunk: int = 512,   # 메모리 안정화용(기능 동일)
    count_entire: bool = False,     # 전체 포인트클라우드로 inlier counting 및 chamfer 계산 여부 (False면 correspondence만)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    2pair 초기 가설 -> inlier counting -> (유효한 모든 가설) 배치 symmetry refinement -> chamfer 필터.
    반환: axes (K,3), angles (K,), chamfers (K,), axis_points (K,3) 축 위 한 점
    """
    n_corr = int(base_corr_np.shape[0])
    if n_corr < seed_pairs:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    P_all = torch.from_numpy(base_corr_np).to(device=device, dtype=torch.float32)      # (N,3)
    Q_all = torch.from_numpy(rotated_corr_np).to(device=device, dtype=torch.float32)  # (N,3)

    thr = torch.tensor(inlier_dist_threshold, device=device, dtype=torch.float32)

    pc = point_cloud
    M = pc.shape[0]
    pc_t = pc.t().contiguous()  # (3,M)

    num_batches = (num_iterations + batch_hyp - 1) // batch_hyp

    axes_list, angles_list, chamfers_list, axis_points_list = [], [], [], []

    # 사전 준비
    P_all_t = P_all.t().contiguous()  # (3,N)
    Q_all_t = Q_all.t().contiguous()  # (3,N)
    P_all_b = P_all.unsqueeze(0)      # (1,N,3)
    Q_all_b = Q_all.unsqueeze(0)      # (1,N,3)

    for _ in range(num_batches):
        # ---- 2pair 샘플링 ----
        idx = torch.randint(0, n_corr, (batch_hyp, seed_pairs), device=device)
        P_seed = P_all[idx]  # (B,2,3)
        Q_seed = Q_all[idx]  # (B,2,3)

        n, p, theta, R, t, valid = _hypothesize_symmetry_from_2pairs_batched(P_seed, Q_seed)
        if not valid.any():
            continue

        # ---- 초기 inlier count (forward/backward min) ----
        if count_entire:
            pcd_rotated = (R @ pc_t).transpose(1, 2) + t.unsqueeze(1)  # (B,M,3)
            pcd_inlier_matrix = torch.cdist(pc, pcd_rotated, p=2)
            inlier_count = (pcd_inlier_matrix < thr).sum(dim=(1, 2)).to(torch.int32)  # (B,)
        else:
            RP = (R @ P_all_t).transpose(1, 2) + t.unsqueeze(1)  # (B,N,3)
            resid_f = torch.linalg.norm(RP - Q_all_b.expand(RP.shape[0], -1, -1), dim=2)  # (B,N)

            Rinv = R.transpose(1, 2)
            tinv = -(Rinv @ t.unsqueeze(-1)).squeeze(-1)
            RQ = (Rinv @ Q_all_t).transpose(1, 2) + tinv.unsqueeze(1)
            resid_b = torch.linalg.norm(RQ - P_all_b.expand(RQ.shape[0], -1, -1), dim=2)

            resid = torch.minimum(resid_f, resid_b)

            inlier_count = (resid < thr).sum(dim=1).to(torch.int32)
        inlier_count = torch.where(valid, inlier_count, torch.full_like(inlier_count, -1))

        # ---- top-k 제거: 유효한 모든 가설 사용 ----
        cand_mask = (inlier_count >= seed_pairs)
        if not cand_mask.any():
            continue
        cand_idx = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)  # (Kcand,)

        # 메모리 안정화를 위해 chunk 처리(기능 동일)
        Kcand = int(cand_idx.numel())
        for s in range(0, Kcand, refine_chunk):
            e = min(s + refine_chunk, Kcand)
            sel = cand_idx[s:e]

            n0 = n[sel]
            p0 = p[sel]
            th0 = theta[sel]

            n_ref, p_ref, th_ref, inl_ref = refine_symmetry_batched_with_weights(
                P_all, Q_all, n0, p0, th0,
                thr=inlier_dist_threshold,
                iters=refine_iters,
                soft_weights=soft_weights,
            )

            keep = th_ref.abs() > identity_theta_thr
            if min_inliers_after_refine is not None:
                keep = keep & (inl_ref >= int(min_inliers_after_refine))
            if min_inlier_ratio_after_refine is not None:
                keep = keep & (inl_ref.float() / float(n_corr) >= float(min_inlier_ratio_after_refine))

            if not keep.any():
                continue

            n_ref = n_ref[keep]
            th_ref = th_ref[keep]
            p_ref = p_ref[keep]
            Kp = n_ref.shape[0]

            # ---- R,t 재구성 ----
            R_ref = _rodrigues_batched(n_ref, th_ref)  # (Kp,3,3)
            t_ref = p_ref - (R_ref @ p_ref.unsqueeze(-1)).squeeze(-1)  # (Kp,3)

            # ---- 배치 chamfer (RX+t) ----
            pc_rep = pc.unsqueeze(0).expand(Kp, -1, -1)  # (Kp,M,3)
            pc_rot = (R_ref @ pc_t.unsqueeze(0).expand(Kp, 3, M)).transpose(1, 2) + t_ref.unsqueeze(1)  # (Kp,M,3)
            ch = _chamfer_batched_knn(pc_rep, pc_rot)  # (Kp,)

            keep2 = ch < chamfer_threshold
            if keep2.any():
                axes_list.append(n_ref[keep2].detach().cpu())
                angles_list.append(th_ref[keep2].detach().cpu())
                chamfers_list.append(ch[keep2].detach().cpu())
                axis_points_list.append(p_ref[keep2].detach().cpu())

    if len(axes_list) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    axes = torch.cat(axes_list, dim=0).numpy().astype(np.float32)
    angles = torch.cat(angles_list, dim=0).numpy().astype(np.float32)
    chamfers = torch.cat(chamfers_list, dim=0).numpy().astype(np.float32)
    axis_points = torch.cat(axis_points_list, dim=0).numpy().astype(np.float32)
    return axes, angles, chamfers, axis_points


# ============================================================
#  Fast Rotation Hypothesis Extraction (Symmetry + Adaptive Iterations)
#  스타일: 기존 _rotations_from_pairs_gpu + adaptive num_iterations
# ============================================================

@torch.no_grad()
def _rotations_from_pairs_gpu_fast(
    base_corr_np: np.ndarray,
    rotated_corr_np: np.ndarray,
    point_cloud: torch.Tensor,          # (M,3) in SAME coordinates as correspondences
    device: torch.device,
    inlier_dist_threshold: float = 0.007,
    num_iterations: int = 20000,
    seed_pairs: int = 2,
    batch_hyp: int = 2048,
    refine_iters: int = 1,
    chamfer_threshold: float = 0.005,
    identity_theta_thr: float = 1e-2,
    soft_weights: bool = False,
    min_inliers_after_refine: int = 10,
    min_inlier_ratio_after_refine: float = 0.00,
    refine_chunk: int = 512,
    count_entire: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive iterations를 적용한 빠른 symmetry-aware rotation 가설 추출.
    
    기존 _rotations_from_pairs_gpu와 동일한 기능을 수행하되:
    - num_iterations를 correspondence 수에 따라 adaptive하게 조정
    - 속도 최적화를 위한 몇 가지 개선 (rotation_RANSAC_JH.py 참고)
    
    반환: axes (K,3), angles (K,), chamfers (K,), axis_points (K,3)
    """
    n_corr = int(base_corr_np.shape[0])
    if n_corr < seed_pairs:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    # ---- Adaptive num_iterations ----
    # delta = 0.05: outlier-free subset을 선택할 확률 95%
    delta = 0.05
    if n_corr > seed_pairs:
        effective_num_iterations = int(np.ceil((n_corr / seed_pairs) * np.log(n_corr / delta)))
    else:
        effective_num_iterations = num_iterations

    print(f"Effective num_iterations: {effective_num_iterations} (n_corr={n_corr}, seed_pairs={seed_pairs})")
    P_all = torch.from_numpy(base_corr_np).to(device=device, dtype=torch.float32)      # (N,3)
    Q_all = torch.from_numpy(rotated_corr_np).to(device=device, dtype=torch.float32)  # (N,3)

    thr = torch.tensor(inlier_dist_threshold, device=device, dtype=torch.float32)

    pc = point_cloud
    M = pc.shape[0]
    pc_t = pc.t().contiguous()  # (3,M)

    num_batches = (effective_num_iterations + batch_hyp - 1) // batch_hyp

    axes_list, angles_list, chamfers_list, axis_points_list = [], [], [], []

    # 사전 준비 (미리 계산해서 반복문 내 오버헤드 감소)
    P_all_t = P_all.t().contiguous()  # (3,N)
    Q_all_t = Q_all.t().contiguous()  # (3,N)
    P_all_b = P_all.unsqueeze(0)      # (1,N,3)
    Q_all_b = Q_all.unsqueeze(0)      # (1,N,3)

    for _ in range(num_batches):
        # ---- 2pair 샘플링 ----
        idx = torch.randint(0, n_corr, (batch_hyp, seed_pairs), device=device)
        P_seed = P_all[idx]  # (B,2,3)
        Q_seed = Q_all[idx]  # (B,2,3)

        n, p, theta, R, t, valid = _hypothesize_symmetry_from_2pairs_batched(P_seed, Q_seed)
        if not valid.any():
            continue

        # ---- 초기 inlier count (forward/backward min) ----
        if count_entire:
            pcd_rotated = (R @ pc_t).transpose(1, 2) + t.unsqueeze(1)  # (B,M,3)
            pcd_inlier_matrix = torch.cdist(pc, pcd_rotated, p=2)
            inlier_count = (pcd_inlier_matrix < thr).sum(dim=(1, 2)).to(torch.int32)  # (B,)
        else:
            RP = (R @ P_all_t).transpose(1, 2) + t.unsqueeze(1)  # (B,N,3)
            resid_f = torch.linalg.norm(RP - Q_all_b.expand(RP.shape[0], -1, -1), dim=2)  # (B,N)

            Rinv = R.transpose(1, 2)
            tinv = -(Rinv @ t.unsqueeze(-1)).squeeze(-1)
            RQ = (Rinv @ Q_all_t).transpose(1, 2) + tinv.unsqueeze(1)
            resid_b = torch.linalg.norm(RQ - P_all_b.expand(RQ.shape[0], -1, -1), dim=2)

            resid = torch.minimum(resid_f, resid_b)

            inlier_count = (resid < thr).sum(dim=1).to(torch.int32)
        inlier_count = torch.where(valid, inlier_count, torch.full_like(inlier_count, -1))

        # ---- 유효한 모든 가설 사용 ----
        cand_mask = (inlier_count >= seed_pairs)
        if not cand_mask.any():
            continue
        cand_idx = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)  # (Kcand,)

        # 메모리 안정화를 위해 chunk 처리
        Kcand = int(cand_idx.numel())
        for s in range(0, Kcand, refine_chunk):
            e = min(s + refine_chunk, Kcand)
            sel = cand_idx[s:e]

            n0 = n[sel]
            p0 = p[sel]
            th0 = theta[sel]

            n_ref, p_ref, th_ref, inl_ref = refine_symmetry_batched_with_weights(
                P_all, Q_all, n0, p0, th0,
                thr=inlier_dist_threshold,
                iters=refine_iters,
                soft_weights=soft_weights,
            )

            keep = th_ref.abs() > identity_theta_thr
            if min_inliers_after_refine is not None:
                keep = keep & (inl_ref >= int(min_inliers_after_refine))
            if min_inlier_ratio_after_refine is not None:
                keep = keep & (inl_ref.float() / float(n_corr) >= float(min_inlier_ratio_after_refine))

            if not keep.any():
                continue

            n_ref = n_ref[keep]
            th_ref = th_ref[keep]
            p_ref = p_ref[keep]
            Kp = n_ref.shape[0]

            # ---- R,t 재구성 ----
            R_ref = _rodrigues_batched(n_ref, th_ref)  # (Kp,3,3)
            t_ref = p_ref - (R_ref @ p_ref.unsqueeze(-1)).squeeze(-1)  # (Kp,3)

            # ---- 배치 chamfer (RX+t) ----
            pc_rep = pc.unsqueeze(0).expand(Kp, -1, -1)  # (Kp,M,3)
            pc_rot = (R_ref @ pc_t.unsqueeze(0).expand(Kp, 3, M)).transpose(1, 2) + t_ref.unsqueeze(1)  # (Kp,M,3)
            ch = _chamfer_batched_knn(pc_rep, pc_rot)  # (Kp,)

            keep2 = ch < chamfer_threshold
            if keep2.any():
                axes_list.append(n_ref[keep2].detach().cpu())
                angles_list.append(th_ref[keep2].detach().cpu())
                chamfers_list.append(ch[keep2].detach().cpu())
                axis_points_list.append(p_ref[keep2].detach().cpu())

    if len(axes_list) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    axes = torch.cat(axes_list, dim=0).numpy().astype(np.float32)
    angles = torch.cat(angles_list, dim=0).numpy().astype(np.float32)
    chamfers = torch.cat(chamfers_list, dim=0).numpy().astype(np.float32)
    axis_points = torch.cat(axis_points_list, dim=0).numpy().astype(np.float32)
    return axes, angles, chamfers, axis_points


# ============================================================
#  DCP Inference Wrapper (correspondence extraction)
# ============================================================

class DCPInference:
    """
    DCP 모델을 사용한 inference wrapper
    Correspondence를 추출할 수 있도록 확장
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        args=None,
        emb_nn: str = "dgcnn",
        pointer: str = "transformer",
        head: str = "svd",
        emb_dims: int = 2048,
        topk: int = 30,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if args is None:
            class Args:
                def __init__(self):
                    self.emb_nn = emb_nn
                    self.pointer = pointer
                    self.head = head
                    self.emb_dims = emb_dims
                    self.n_blocks = 1
                    self.n_heads = 4
                    self.ff_dims = 1024
                    self.dropout = 0.0
                    self.cycle = False
            args = Args()

        self.args = args
        self.model = DCP(args).to(self.device)
        self.model.eval()

        self.topk = topk

        # resolve model path
        if not os.path.isabs(model_path):
            if os.path.isfile(model_path):
                model_path = os.path.abspath(model_path)
            elif os.path.isfile(osp.join(osp.dirname(__file__), model_path)):
                model_path = osp.join(osp.dirname(__file__), model_path)
            elif os.path.isfile(osp.join(osp.dirname(__file__), "pretrained", os.path.basename(model_path))):
                model_path = osp.join(osp.dirname(__file__), "pretrained", os.path.basename(model_path))

        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded DCP weights from {model_path}")
        else:
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please check the path. Available models in pretrained/: "
                f"{os.listdir(osp.join(osp.dirname(__file__), 'pretrained')) if osp.exists(osp.join(osp.dirname(__file__), 'pretrained')) else 'N/A'}"
            )

    @torch.no_grad()
    def get_correspondences(
        self,
        src_points: np.ndarray,
        tgt_points: np.ndarray,
        return_scores: bool = False,
    ) -> tuple:
        src_tensor = torch.from_numpy(src_points.astype(np.float32)).T.unsqueeze(0).to(self.device)  # (1,3,N)
        tgt_tensor = torch.from_numpy(tgt_points.astype(np.float32)).T.unsqueeze(0).to(self.device)  # (1,3,M)

        src_embedding = self.model.emb_nn(src_tensor)
        tgt_embedding = self.model.emb_nn(tgt_tensor)

        src_embedding_p, tgt_embedding_p = self.model.pointer(src_embedding, tgt_embedding)
        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / np.sqrt(d_k)  # (1,N,M)

        # geometric prefilter (자기대응에서 너무 가까운 점 매칭 억제)
        src_pts = src_tensor.squeeze(0).transpose(0, 1)  # (N,3)
        tgt_pts = tgt_tensor.squeeze(0).transpose(0, 1)  # (M,3)
        dists = torch.norm(src_pts.unsqueeze(1) - tgt_pts.unsqueeze(0), dim=2)  # (N,M)
        min_radius = 0.1
        scores = scores.masked_fill(~(dists >= min_radius).unsqueeze(0), -1e9)

        scores = torch.softmax(scores, dim=2).squeeze(0)  # (N,M)

        topk = self.topk
        _, src_topk_indices = scores.topk(k=topk, dim=1)
        src_corr_mat = torch.zeros_like(scores, dtype=torch.bool)
        src_indices = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1).expand(-1, topk)
        src_corr_mat[src_indices, src_topk_indices] = True

        _, tgt_topk_indices = scores.topk(k=topk, dim=0)
        tgt_corr_mat = torch.zeros_like(scores, dtype=torch.bool)
        tgt_indices = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).expand(topk, -1)
        tgt_corr_mat[tgt_topk_indices, tgt_indices] = True

        mutual = torch.logical_and(src_corr_mat, tgt_corr_mat)
        si, ti = torch.nonzero(mutual, as_tuple=True)
        conf = scores[si, ti]

        min_confidence = 0.005
        keep = conf > min_confidence
        si, ti = si[keep], ti[keep]

        src_np = src_tensor.squeeze(0).T.cpu().numpy()  # (N,3)
        tgt_np = tgt_tensor.squeeze(0).T.cpu().numpy()  # (M,3)

        src_corr = src_np[si.cpu().numpy()]
        tgt_corr = tgt_np[ti.cpu().numpy()]

        if return_scores:
            return src_corr, tgt_corr, scores.cpu().numpy()
        return src_corr, tgt_corr

    @torch.no_grad()
    def get_transformation(self, src_points: np.ndarray, tgt_points: np.ndarray) -> tuple:
        src_tensor = torch.from_numpy(src_points.astype(np.float32)).T.unsqueeze(0).to(self.device)
        tgt_tensor = torch.from_numpy(tgt_points.astype(np.float32)).T.unsqueeze(0).to(self.device)
        rotation, translation, _, _ = self.model(src_tensor, tgt_tensor)
        return rotation.squeeze(0).cpu().numpy(), translation.squeeze(0).cpu().numpy()


def get_correspondences(
    inference_model: DCPInference,
    point_cloud: np.ndarray,
    rotation_matrix: np.ndarray,
    return_scores: bool = False,
) -> tuple:
    # symmetry detection setting: base와 rotated 동일 cloud로 self-correspondence 사용
    base = point_cloud.astype(np.float32)
    rotated = base.copy()

    if return_scores:
        base_corr, rotated_corr, scores = inference_model.get_correspondences(base, rotated, return_scores=True)
        return base, rotated, base_corr, rotated_corr, scores
    else:
        base_corr, rotated_corr = inference_model.get_correspondences(base, rotated, return_scores=False)
        return base, rotated, base_corr, rotated_corr


# ============================================================
#  Evaluation
# ============================================================

@torch.no_grad()
def evaluate_rotation_symmetry_by_correspondence(
    point_clouds: np.ndarray,
    gt_labels: List[List[np.ndarray]],
    inference_model: DCPInference,
    dist_threshold: float = 0.05,
    chamfer_threshold: float = 0.005,
    count_entire: bool = False,
) -> Tuple[float, float, float, float]:
    """
    ShapeNet rotation symmetry 평가.
    TP: (1) normal 차이 < dist_threshold, (2) 예측축 직선(점 p+방향 n)과 GT축 직선(원점+gt_axis) 사이 거리 < dist_threshold.
    """
    device = inference_model.device
    precision_sum = 0.0
    recall_sum = 0.0
    low_chamfer_sum = 0.0
    low_chamfer_count = 0
    total_samples = len(point_clouds)

    gt_lines: List[str] = []
    out_gt_path = osp.join(osp.dirname(__file__), "dcp_rotation_symmetry_shapenet.txt")

    pred_axis_labels: List[str] = []
    pred_point_labels: List[str] = []
    pbar = tqdm(range(total_samples), desc="Evaluating samples", dynamic_ncols=True)
    for i in pbar:
        pc = point_clouds[i].astype(np.float32)

        gt_params_list = gt_labels[i]
        if len(gt_params_list) == 0:
            # GT rotation이 없는 샘플은 skip (원본 코드 유지)
            precision_sum += 1.0
            continue

        gt_axes_list = []
        for p in gt_params_list:
            p_arr = np.asarray(p).reshape(-1)
            if p_arr.shape[0] >= 3:
                gt_axes_list.append(p_arr[:3])

        if len(gt_axes_list) == 0:
            precision_sum += 1.0
            continue

        gt_axes_tensor = torch.stack(
            [torch.tensor(a, dtype=torch.float32, device=device) for a in gt_axes_list]
        )
        gt_axes_tensor = gt_axes_tensor / (torch.norm(gt_axes_tensor, dim=1, keepdim=True) + 1e-8)

        rotations_kept = 0

        base = pc.astype(np.float32)
        _, _, base_corr_all, rotated_corr_all = get_correspondences(
            inference_model, base, np.eye(3, dtype=np.float32)
        )

        if rotated_corr_all.shape[0] == 0:
            base_corr_all = np.zeros((0, 3), dtype=np.float32)
            rotated_corr_all = np.zeros((0, 3), dtype=np.float32)

        num_correspondences = base_corr_all.shape[0]

        # IMPORTANT: point_cloud는 correspondences와 같은 좌표계(uncentered)로 전달
        X = torch.from_numpy(base).float().to(device)

        # --------- 2pair symmetry RANSAC + constraint refinement ----------
        axes_np, angles_np, chamfer_np, axis_points_np = _rotations_from_pairs_gpu(
            base_corr_all,
            rotated_corr_all,
            X,
            device,
            inlier_dist_threshold=0.007,
            num_iterations=30000,
            seed_pairs=2,
            batch_hyp=256,
            refine_iters=1,
            chamfer_threshold=chamfer_threshold,
            identity_theta_thr=2e-1,
            count_entire=count_entire,
        )

        keep_axes = None
        keep_angles = None
        keep_chamfers = None
        keep_axis_points = None

        if axes_np.shape[0] > 0:
            axes_batch = torch.from_numpy(axes_np).to(device)
            angles_batch = torch.from_numpy(angles_np).to(device, dtype=torch.float32)
            chamfer_batch = torch.from_numpy(chamfer_np).to(device, dtype=torch.float32)
            axis_points_batch = torch.from_numpy(axis_points_np).to(device, dtype=torch.float32)

            # 1) chamfer 필터
            chamfer_mask = chamfer_batch < chamfer_threshold

            if chamfer_mask.any():
                # 2) non-identity 필터
                identity_angle_thr_eval = 1e-2  # rad ~0.57°
                non_identity_mask = angles_batch.abs() > identity_angle_thr_eval
                final_mask = chamfer_mask & non_identity_mask

                if final_mask.any():
                    keep_axes = axes_batch[final_mask]
                    keep_angles = angles_batch[final_mask]
                    keep_chamfers = chamfer_batch[final_mask]
                    keep_axis_points = axis_points_batch[final_mask]

        if keep_axes is None or keep_axes.shape[0] == 0:
            precision_sum += 1.0
        else:
            # 3) axis NMS (chamfer 작은 것 우선)
            axis_merge_threshold = 0.2
            order = torch.argsort(keep_chamfers)  # chamfer 오름차순

            selected_indices: List[int] = []
            for idx in order.tolist():
                a = keep_axes[idx]
                if len(selected_indices) == 0:
                    selected_indices.append(idx)
                    continue

                sel_axes = keep_axes[selected_indices]
                dist_pos = torch.norm(sel_axes - a.unsqueeze(0), dim=1)
                dist_neg = torch.norm(sel_axes + a.unsqueeze(0), dim=1)
                axis_dist = torch.minimum(dist_pos, dist_neg)

                if axis_dist.min().item() >= axis_merge_threshold:
                    selected_indices.append(idx)

            keep_axes = keep_axes[selected_indices]
            keep_angles = keep_angles[selected_indices]
            keep_chamfers = keep_chamfers[selected_indices]
            keep_axis_points = keep_axis_points[selected_indices]
            rotations_kept = keep_axes.shape[0]

            low_chamfer_sum += float(keep_chamfers.sum().item())
            low_chamfer_count += len(keep_chamfers)

            # TP: (1) normal 차이 < dist_threshold (0.05), (2) 축 사이 직선거리 < dist_threshold (0.05)
            pred_axes_expanded = keep_axes.unsqueeze(1)   # (K,1,3)
            gt_axes_expanded = gt_axes_tensor.unsqueeze(0)  # (1,G,3)
            dist_pos = torch.norm(pred_axes_expanded - gt_axes_expanded, dim=2)
            dist_neg = torch.norm(pred_axes_expanded + gt_axes_expanded, dim=2)
            axis_dist = torch.minimum(dist_pos, dist_neg)  # (K,G) normal 차이

            line_dist = _line_to_line_distance_batched(
                keep_axis_points, keep_axes, gt_axes_tensor
            )  # (K,G) 직선(원점+gt)과 직선(p_pred+n_pred) 사이 거리

            match_matrix = (axis_dist < dist_threshold) & (line_dist < 0.1)

            matches_per_pred = match_matrix.any(dim=1)
            tp_precision = int(matches_per_pred.sum().item())
            precision_sum += tp_precision / keep_axes.shape[0]

            matches_per_gt = match_matrix.any(dim=0)
            tp_recall = int(matches_per_gt.sum().item())
            recall_sum += tp_recall / len(gt_axes_tensor)

            min_gt_dist_per_pred = axis_dist.min(dim=1)[0]
            for k in range(keep_axes.shape[0]):
                axis = keep_axes[k]
                angle = keep_angles[k].item()
                chamfer_val = keep_chamfers[k].item()
                min_metric = min_gt_dist_per_pred[k].item()
                gt_lines.append(
                    f"{i}\t{float(axis[0]):.6f}\t{float(axis[1]):.6f}\t{float(axis[2]):.6f}\t"
                    f"{angle:.6f}\t{chamfer_val:.6f}\t{min_metric:.6f}"
                )

        ## for visualization
        if axes_np.shape[0] > 0 and chamfer_mask.any() and final_mask.any():
            final_mask_np = final_mask.detach().cpu().numpy().astype(bool).reshape(-1)
            pred_axis_labels.append(axes_np[final_mask_np][selected_indices].reshape(-1, 3))
            pred_point_labels.append(axis_points_np[final_mask_np][selected_indices].reshape(-1, 3))
        else:
            pred_axis_labels.append(np.empty((0, 3), dtype=np.float32))
            pred_point_labels.append(np.empty((0, 3), dtype=np.float32))

        done = i + 1
        pbar.set_postfix(
            corr=num_correspondences,
            rotations=rotations_kept,
            P=(precision_sum / done) if done > 0 else 0.0,
            R=(recall_sum / done) if done > 0 else 0.0,
        )

    try:
        with open(out_gt_path, "w") as f:
            f.write("idx\tax\tay\taz\tangle\tchamfer\tmin_gt_dist\n")
            for line in gt_lines:
                f.write(line + "\n")
    except Exception:
        pass

    precision = precision_sum / total_samples if total_samples > 0 else 0.0
    recall = recall_sum / total_samples if total_samples > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_chamfer = low_chamfer_sum / low_chamfer_count if low_chamfer_count > 0 else 0.0
    return precision, recall, f1, avg_chamfer, pred_axis_labels, pred_point_labels


# ============================================================
#  Cache / Path helpers and main
# ============================================================

def _resolve_mesh_root(mesh_root_arg: Optional[str], script_dir: str, repo_root: str) -> Optional[str]:
    """환경변수·기본 경로 후보를 순서대로 확인해 존재하는 디렉터리 반환."""
    candidates = []
    if mesh_root_arg and mesh_root_arg.strip():
        candidates.append(osp.abspath(mesh_root_arg.strip()))
    if os.environ.get("SHAPENET_ROOT"):
        candidates.append(osp.abspath(os.environ["SHAPENET_ROOT"]))
    if os.environ.get("MESH_ROOT"):
        candidates.append(osp.abspath(os.environ["MESH_ROOT"]))
    candidates.extend([
        osp.join(repo_root, "ShapeNetCore.v2"),
        osp.join(repo_root, "..", "ShapeNetCore.v2"),
        osp.join(script_dir, "ShapeNetCore.v2"),
        osp.abspath(osp.join(os.getcwd(), "ShapeNetCore.v2")),
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


def main():
    # Optional perf toggles (safe)
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    script_dir = osp.dirname(osp.abspath(__file__))
    repo_root = osp.abspath(osp.join(script_dir, ".."))

    parser = argparse.ArgumentParser(description="DCP Rotation Symmetry Evaluation on ShapeNet (2pair + symmetry refinement)")
    default_model = None
    pretrained_dir = osp.join(osp.dirname(__file__), "pretrained")
    if osp.exists(pretrained_dir):
        if osp.exists(osp.join(pretrained_dir, "dcp_v2.t7")):
            default_model = osp.join(pretrained_dir, "dcp_v2.t7")
        elif osp.exists(osp.join(pretrained_dir, "dcp_v1.t7")):
            default_model = osp.join(pretrained_dir, "dcp_v1.t7")

    parser.add_argument("--model_path", type=str, default=default_model,
                        help=f"Path to DCP model weights (default: {default_model if default_model else 'required'})")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points per point cloud")
    parser.add_argument("--dist_threshold", type=float, default=0.05, help="Axis matching distance threshold")
    parser.add_argument("--chamfer_threshold", type=float, default=0.0004, help="Chamfer distance threshold")
    parser.add_argument("--emb_nn", type=str, default="dgcnn", choices=["pointnet", "dgcnn"], help="Embedding network type")
    parser.add_argument("--pointer", type=str, default="transformer", choices=["identity", "transformer"], help="Pointer type")
    parser.add_argument("--head", type=str, default="svd", choices=["mlp", "svd"], help="Head type")
    parser.add_argument("--emb_dims", type=int, default=512, help="Embedding dimensions")
    
    parser.add_argument("--count_entire", action="store_true", help="Count all valid hypotheses for whole point cloud (default: only correspondences)")
    parser.add_argument("--topk", type=int, default=60, help="Top-k correspondences to consider for symmetry detection (default: 30)")

    parser.add_argument("--cache_file", type=str, default=None,
                        help="Pre-built pkl (point_clouds + gt_labels). Default: dcp/shapenet_rot_gt_cache.pkl")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="model_symmetry 폴더 있는 경로 (캐시 없을 때만). Default: dcp/model_symmetry_extracted")
    parser.add_argument("--mesh_root", type=str, default=None,
                        help="ShapeNetCore.v2 경로 (캐시 없을 때만). Default: env 또는 repo/dcp 내 ShapeNetCore.v2")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="최대 샘플 수 (None = 전부). 캐시 빌드/평가 둘 다 적용")
    parser.add_argument("--visualization", action="store_true", help="Save visualization of predictions and GTs")

    args = parser.parse_args()

    if args.model_path is None:
        parser.error("--model_path is required. Put dcp_v1.t7 or dcp_v2.t7 in dcp/pretrained/ or set --model_path")

    cache_path = args.cache_file or osp.join(script_dir, "data", "shapenet_rot_gt_cache.pkl")

    # cache load/build
    if osp.isfile(cache_path):
        print(f"Loading cache: {cache_path}")
        import pickle
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        point_clouds = data["point_clouds"]
        gt_labels = data["gt_labels"]
        if args.max_samples is not None and len(point_clouds) > args.max_samples:
            point_clouds = point_clouds[:args.max_samples]
            gt_labels = gt_labels[:args.max_samples]
        print(f"Loaded {len(point_clouds)} samples from cache")
    else:
        if args.dataset_root is None:
            args.dataset_root = osp.join(script_dir, "model_symmetry_extracted")
        if not osp.isdir(args.dataset_root) or not osp.isdir(osp.join(args.dataset_root, "model_symmetry")):
            print(f"Error: dataset_root (model_symmetry) not found: {args.dataset_root}")
            print("  Extract model_symmetry.zip to dcp/model_symmetry_extracted/ or set --dataset_root.")
            sys.exit(1)

        args.mesh_root = _resolve_mesh_root(args.mesh_root, script_dir, repo_root)
        if not args.mesh_root:
            print("Error: ShapeNet mesh root not found. Set --mesh_root or put ShapeNetCore.v2 in repo/dcp/.")
            sys.exit(1)

        print("Cache not found. Building from model_symmetry + ShapeNetCore.v2...")
        try:
            point_clouds, gt_labels = build_shapenet_rot_cache(
                args.dataset_root,
                args.mesh_root,
                cache_path,
                num_points=args.num_points,
                max_samples=args.max_samples,
            )
        except Exception as e:
            print(f"Error building cache: {e}")
            sys.exit(1)

        if args.max_samples is not None and len(point_clouds) > args.max_samples:
            point_clouds = point_clouds[:args.max_samples]
            gt_labels = gt_labels[:args.max_samples]

    if len(point_clouds) == 0:
        print("Error: No samples in cache or built. Check dataset_root and mesh_root.")
        sys.exit(1)

    infer = DCPInference(
        args.model_path,
        device=args.device,
        emb_nn=args.emb_nn,
        pointer=args.pointer,
        head=args.head,
        emb_dims=args.emb_dims,
        topk=args.topk,
    )

    precision, recall, f1, avg_ch, pred_axis_labels, pred_point_labels = evaluate_rotation_symmetry_by_correspondence(
        point_clouds,
        gt_labels,
        infer,
        dist_threshold=args.dist_threshold,
        chamfer_threshold=args.chamfer_threshold,
        count_entire=args.count_entire,
    )

    if args.visualization:
        vis_dir = osp.join(script_dir, "vis_qual", "rotation")
        from vis_qual.visualization_html_rotation import save_visualization
        axis_for_vis = np.stack([
            np.vstack([np.asarray(a).reshape(-1)[:3] for a in sample])
            for sample in gt_labels
        ]) 
        save_visualization(point_clouds, axis_for_vis, np.zeros_like(axis_for_vis), vis_dir, prefix="gt")
        save_visualization(point_clouds, pred_axis_labels, pred_point_labels, vis_dir, prefix="pred")

    print("\n=== Evaluation Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Avg Chamfer: {avg_ch:.6f}")
    print(f"\nEvaluated {len(point_clouds)} samples with rotation symmetry GT")


if __name__ == "__main__":
    main()