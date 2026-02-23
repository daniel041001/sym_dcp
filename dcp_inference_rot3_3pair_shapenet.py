"""
DCP 모델을 사용하여 correspondence를 추출하는 inference 함수
ShapeNet 데이터셋에서 rotation symmetry detection 평가

- SymmetryNet model_sym.txt GT (model_symmetry/<class_id>/<ins_name>/model_sym.txt)
- 포인트 클라우드: 원본 ShapeNet 메시에서 trimesh 로 샘플링 (symmetrynet_shapenet_gt.load_point_cloud_from_mesh)
"""

import os
import os.path as osp
import sys

# dcp/ 에서 직접 실행해도 import 가능하도록 repo root 를 path 에 추가
_script_dir = osp.dirname(osp.abspath(__file__))
_repo_root = osp.abspath(osp.join(_script_dir, '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import argparse
from typing import List, Tuple, Optional
import torch
import numpy as np
from pytorch3d.loss import chamfer_distance
from model import DCP

# SymmetryNet ShapeNet GT 로더 (원본 메시 → 포인트클라우드는 trimesh)
from dcp.symmetrynet_shapenet_gt import (
    build_shapenet_rot_cache,
    get_shapenet_rotation_gt,
    load_point_cloud_from_mesh,
)

# tqdm (with safe fallback)
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


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


class DCPInference:
    """
    DCP 모델을 사용한 inference wrapper
    Correspondence를 추출할 수 있도록 확장
    """

    def __init__(self, model_path: str, device: str = 'cuda:0', args=None,
                 emb_nn: str = 'dgcnn', pointer: str = 'transformer',
                 head: str = 'svd', emb_dims: int = 2048):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

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

        if not os.path.isabs(model_path):
            if os.path.isfile(model_path):
                model_path = os.path.abspath(model_path)
            elif os.path.isfile(osp.join(osp.dirname(__file__), model_path)):
                model_path = osp.join(osp.dirname(__file__), model_path)
            elif os.path.isfile(osp.join(osp.dirname(__file__), 'pretrained', os.path.basename(model_path))):
                model_path = osp.join(osp.dirname(__file__), 'pretrained', os.path.basename(model_path))

        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'], strict=False)
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
        return_scores: bool = False
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

        # geometric prefilter
        src_pts = src_tensor.squeeze(0).transpose(0, 1)  # (N,3)
        tgt_pts = tgt_tensor.squeeze(0).transpose(0, 1)  # (M,3)
        dists = torch.norm(src_pts.unsqueeze(1) - tgt_pts.unsqueeze(0), dim=2)  # (N,M)
        min_radius = 0.1
        scores = scores.masked_fill(~(dists >= min_radius).unsqueeze(0), -1e9)

        scores = torch.softmax(scores, dim=2).squeeze(0)  # (N,M)

        topk = 40

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

        src_np = src_tensor.squeeze(0).T.cpu().numpy()
        tgt_np = tgt_tensor.squeeze(0).T.cpu().numpy()

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
    return_scores: bool = False
) -> tuple:
    # NOTE: symmetry detection setting에서는 base와 rotated를 동일 cloud로 두고,
    #       self-correspondence로부터 candidate rotation을 RANSAC으로 추정하는 구조를 유지.
    base = point_cloud.astype(np.float32)
    rotated = base.copy()

    if return_scores:
        base_corr, rotated_corr, scores = inference_model.get_correspondences(base, rotated, return_scores=True)
        return base, rotated, base_corr, rotated_corr, scores
    else:
        base_corr, rotated_corr = inference_model.get_correspondences(base, rotated, return_scores=False)
        return base, rotated, base_corr, rotated_corr


@torch.no_grad()
def evaluate_rotation_symmetry_by_correspondence(
    point_clouds: np.ndarray,
    gt_labels: List[List[np.ndarray]],
    inference_model: DCPInference,
    dist_threshold: float = 0.05,
    chamfer_threshold: float = 0.01,
    angle_tolerance: float = 0.2,  # ShapeNet은 180°만 지원하므로 각도 허용 오차
) -> Tuple[float, float, float, float]:
    """
    ShapeNet rotation symmetry 평가.
    
    Parameters
    ----------
    point_clouds : (N, M, 3) array
        포인트 클라우드 배열
    gt_labels : list of list of (4,) array
        각 샘플의 GT rotation symmetry 목록 [axis(3) + angle(1)]
        ShapeNet은 180° (π)만 지원
    inference_model : DCPInference
        DCP inference 모델
    dist_threshold : float
        axis matching distance threshold
    chamfer_threshold : float
        chamfer distance threshold for filtering predictions
    angle_tolerance : float
        각도 매칭 허용 오차 (rad). ShapeNet GT는 π이므로 ±angle_tolerance 범위 내면 매칭
    """
    device = inference_model.device
    precision_sum = 0.0
    recall_sum = 0.0
    low_chamfer_sum = 0.0
    low_chamfer_count = 0
    total_samples = len(point_clouds)

    gt_lines: List[str] = []
    out_gt_path = osp.join(osp.dirname(__file__), 'dcp_rotation_symmetry_shapenet.txt')

    pbar = tqdm(range(total_samples), desc='Evaluating samples', dynamic_ncols=True)
    for i in pbar:
        pc = point_clouds[i].astype(np.float32)

        gt_params_list = gt_labels[i]
        if len(gt_params_list) == 0:
            # GT rotation이 없는 샘플은 skip (또는 precision=1.0으로 처리)
            precision_sum += 1.0
            continue

        gt_axes_list = [p[:3] for p in gt_params_list]
        gt_angles_list = [float(p[3]) for p in gt_params_list]

        gt_axes_tensor = torch.stack([torch.tensor(a, dtype=torch.float32, device=device) for a in gt_axes_list])
        gt_axes_tensor = gt_axes_tensor / (torch.norm(gt_axes_tensor, dim=1, keepdim=True) + 1e-8)
        gt_angles_tensor = torch.tensor(gt_angles_list, dtype=torch.float32, device=device)

        pred_axes_torch: List[torch.Tensor] = []
        pred_angles: List[float] = []
        pred_chamfers: List[float] = []
        rotations_kept = 0

        base = pc.astype(np.float32)

        _, _, base_corr_all, rotated_corr_all = get_correspondences(
            inference_model, base, np.eye(3, dtype=np.float32)
        )

        if rotated_corr_all.shape[0] == 0:
            base_corr_all = np.zeros((0, 3), dtype=np.float32)
            rotated_corr_all = np.zeros((0, 3), dtype=np.float32)

        num_correspondences = base_corr_all.shape[0]

        X = torch.from_numpy(base).float().to(device)
        X_orig = X - X.mean(dim=0)

        # --------- GPU batched RANSAC ----------
        axes_np, angles_np, chamfer_np = _rotations_from_pairs_gpu(
            base_corr_all,
            rotated_corr_all,
            X_orig,
            device,
            inlier_dist_threshold=0.01,
            num_iterations=50000,
            seed_pairs=3,
            batch_hyp=256,
            topk_chamfer=8,
        )

        if axes_np.shape[0] > 0:
            axes_batch = torch.from_numpy(axes_np).to(device)
            angles_batch = torch.from_numpy(angles_np).to(device, dtype=torch.float32)
            chamfer_batch = torch.from_numpy(chamfer_np).to(device, dtype=torch.float32)

            filter_mask = chamfer_batch < chamfer_threshold
            if filter_mask.any():
                pred_axes_torch = axes_batch[filter_mask].unbind(0)
                pred_angles = angles_batch[filter_mask].tolist()
                pred_chamfers = chamfer_batch[filter_mask].tolist()

        if len(pred_axes_torch) == 0:
            precision_sum += 1.0
        else:
            keep_axes = torch.stack(pred_axes_torch)
            keep_angles = torch.tensor(pred_angles, dtype=torch.float32, device=device)
            keep_chamfers = torch.tensor(pred_chamfers, dtype=torch.float32, device=device)
            rotations_kept = keep_axes.shape[0]

            low_chamfer_sum += float(keep_chamfers.sum().item())
            low_chamfer_count += len(keep_chamfers)

            pred_axes_expanded = keep_axes.unsqueeze(1)
            gt_axes_expanded = gt_axes_tensor.unsqueeze(0)

            dist_pos = torch.norm(pred_axes_expanded - gt_axes_expanded, dim=2)
            dist_neg = torch.norm(pred_axes_expanded + gt_axes_expanded, dim=2)
            axis_dist = torch.minimum(dist_pos, dist_neg)

            pred_angles_expanded = keep_angles.unsqueeze(1)
            gt_angles_expanded = gt_angles_tensor.unsqueeze(0)

            # ShapeNet은 180° (π)만 지원하므로 각도 매칭을 더 관대하게
            angle_diff = torch.abs(pred_angles_expanded - gt_angles_expanded)
            angle_diff = torch.minimum(angle_diff, 2 * np.pi - angle_diff)
            # π에 가까운 각도도 매칭 (180° ± tolerance)
            angle_diff_from_pi = torch.abs(pred_angles_expanded - np.pi)
            angle_diff_from_pi = torch.minimum(angle_diff_from_pi, 2 * np.pi - angle_diff_from_pi)
            angle_diff = torch.minimum(angle_diff, angle_diff_from_pi)

            angle_weight = 0.1
            metric = axis_dist ## + angle_weight * angle_diff / np.pi

            match_matrix = metric < dist_threshold

            matches_per_pred = match_matrix.any(dim=1)
            tp_precision = int(matches_per_pred.sum().item())
            precision_sum += tp_precision / keep_axes.shape[0]

            matches_per_gt = match_matrix.any(dim=0)
            tp_recall = int(matches_per_gt.sum().item())
            recall_sum += tp_recall / len(gt_axes_tensor)

            min_gt_dist_per_pred = metric.min(dim=1)[0]
            for k in range(keep_axes.shape[0]):
                axis = keep_axes[k]
                angle = keep_angles[k].item()
                chamfer_val = keep_chamfers[k].item()
                min_metric = min_gt_dist_per_pred[k].item()
                gt_lines.append(
                    f"{i}\t{float(axis[0]):.6f}\t{float(axis[1]):.6f}\t{float(axis[2]):.6f}\t"
                    f"{angle:.6f}\t{chamfer_val:.6f}\t{min_metric:.6f}"
                )

        done = i + 1
        pbar.set_postfix(
            corr=num_correspondences,
            rotations=rotations_kept,
            P=(precision_sum / done) if done > 0 else 0.0,
            R=(recall_sum / done) if done > 0 else 0.0,
        )

    try:
        with open(out_gt_path, 'w') as f:
            f.write("idx\tax\tay\taz\tangle\tchamfer\tmin_gt_dist\n")
            for line in gt_lines:
                f.write(line + "\n")
    except Exception:
        pass

    precision = precision_sum / total_samples if total_samples > 0 else 0.0
    recall = recall_sum / total_samples if total_samples > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_chamfer = low_chamfer_sum / low_chamfer_count if low_chamfer_count > 0 else 0.0
    return precision, recall, f1, avg_chamfer


def _resolve_mesh_root(mesh_root_arg: Optional[str], script_dir: str, repo_root: str) -> Optional[str]:
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


def main():
    # Optional perf toggles (safe)
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    script_dir = osp.dirname(osp.abspath(__file__))
    repo_root = osp.abspath(osp.join(script_dir, '..'))

    parser = argparse.ArgumentParser(description='DCP Rotation Symmetry Evaluation on ShapeNet')
    default_model = None
    pretrained_dir = osp.join(osp.dirname(__file__), 'pretrained')
    if osp.exists(pretrained_dir):
        if osp.exists(osp.join(pretrained_dir, 'dcp_v2.t7')):
            default_model = osp.join(pretrained_dir, 'dcp_v2.t7')
        elif osp.exists(osp.join(pretrained_dir, 'dcp_v1.t7')):
            default_model = osp.join(pretrained_dir, 'dcp_v1.t7')

    parser.add_argument('--model_path', type=str, default=default_model,
                        help=f'Path to DCP model weights (default: {default_model if default_model else "required"})')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points per point cloud')
    parser.add_argument('--dist_threshold', type=float, default=0.1, help='Distance threshold for matching')
    parser.add_argument('--chamfer_threshold', type=float, default=0.005, help='Chamfer distance threshold')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Embedding network type')
    parser.add_argument('--pointer', type=str, default='transformer', choices=['identity', 'transformer'], help='Pointer type')
    parser.add_argument('--head', type=str, default='svd', choices=['mlp', 'svd'], help='Head type')
    parser.add_argument('--emb_dims', type=int, default=512, help='Embedding dimensions')

    # ShapeNet: 캐시 파일 있으면 그대로 로드, 없으면 dataset_root + mesh_root 로 빌드 후 저장
    parser.add_argument('--cache_file', type=str, default=None,
                        help='Pre-built pkl (point_clouds + gt_labels). Default: dcp/shapenet_rot_gt_cache.pkl')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='model_symmetry 폴더 있는 경로 (캐시 없을 때만). Default: dcp/model_symmetry_extracted')
    parser.add_argument('--mesh_root', type=str, default=None,
                        help='ShapeNetCore.v2 경로 (캐시 없을 때만). Default: env 또는 repo/dcp 내 ShapeNetCore.v2')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='최대 샘플 수 (None = 전부). 캐시 빌드/평가 둘 다 적용')

    args = parser.parse_args()

    if args.model_path is None:
        parser.error("--model_path is required. Put dcp_v1.t7 or dcp_v2.t7 in dcp/pretrained/ or set --model_path")

    cache_path = args.cache_file or osp.join(script_dir, 'shapenet_rot_gt_cache.pkl')

    # 캐시 있으면 로드, 없으면 dataset_root + mesh_root 로 빌드 후 사용 (name_list.txt 불필요)
    if osp.isfile(cache_path):
        print(f"Loading cache: {cache_path}")
        import pickle
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        point_clouds = data['point_clouds']
        gt_labels = data['gt_labels']
        if args.max_samples is not None and len(point_clouds) > args.max_samples:
            point_clouds = point_clouds[:args.max_samples]
            gt_labels = gt_labels[:args.max_samples]
        print(f"Loaded {len(point_clouds)} samples from cache")
    else:
        if args.dataset_root is None:
            args.dataset_root = osp.join(script_dir, 'model_symmetry_extracted')
        if not osp.isdir(args.dataset_root) or not osp.isdir(osp.join(args.dataset_root, 'model_symmetry')):
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
        emb_dims=args.emb_dims
    )

    precision, recall, f1, avg_ch = evaluate_rotation_symmetry_by_correspondence(
        point_clouds,
        gt_labels,
        infer,
        dist_threshold=args.dist_threshold,
        chamfer_threshold=args.chamfer_threshold,
    )

    print(f"\n=== Evaluation Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Avg Chamfer: {avg_ch:.6f}")
    print(f"\nEvaluated {len(point_clouds)} samples with rotation symmetry GT")


if __name__ == '__main__':
    main()
