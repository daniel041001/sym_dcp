"""
DCP 모델을 사용하여 correspondence를 추출하는 inference 함수
Rotation symmetry detection을 위한 토이 데이터셋 사용
 
요청 반영(이번 수정):
- 정삼각뿔/정사각뿔/정오각뿔/정육각뿔 4종 데이터셋 생성
- 각 물체당 50개씩 총 200개 샘플 생성
- 각 샘플마다 랜덤 SE(3) (회전 + 평행이동) 적용해서 hard case 구성
- 각 샘플에 대해 rotation symmetry GT(월드 좌표 axis + 모든 비자명 각도 2πk/n, k=1..n-1) 생성
"""
 
import os
import os.path as osp
import argparse
from typing import List, Tuple
import torch
import numpy as np
from pytorch3d.loss import chamfer_distance
from model import DCP
 
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
 
 
# -----------------------------
# NEW: multi-shape toy dataset
# -----------------------------
def generate_toy_dataset(
    per_class: int = 50,
    num_points: int = 2048,
    seed: int = 42,
    n_sides_list: List[int] = [3, 4, 5, 6],
    apex_height: float = 1.5,
    trans_range: float = 0.5,
) -> Tuple[np.ndarray, List[List[np.ndarray]], List[int]]:
    """
    Regular n-gonal pyramid dataset (4 classes).
      - n_sides=3: triangular pyramid (정삼각뿔)
      - n_sides=4: square pyramid (정사각뿔)
      - n_sides=5: pentagonal pyramid (정오각뿔)
      - n_sides=6: hexagonal pyramid (정육각뿔)
 
    Each sample: random SE(3) applied (hard case)
    GT: axis in WORLD coords + all non-trivial rotation angles 2πk/n, k=1..n-1
 
    Returns:
      pcs: (total, N, 3)
      labels: list length total, each is list of [axis(3)+angle(1)]
      meta_n: list length total, n_sides per sample
    """
    np.random.seed(seed)
 
    def random_rotation_matrix() -> np.ndarray:
        A = np.random.randn(3, 3).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        return Q.astype(np.float32)
 
    def generate_ngonal_pyramid(n_sides: int, n_points: int) -> np.ndarray:
        # Regular n-gon base on z=0
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False).astype(np.float32) + np.pi / 2
        base_vertices = np.column_stack([
            np.cos(angles),
            np.sin(angles),
            np.zeros(n_sides, dtype=np.float32),
        ]).astype(np.float32)
 
        base_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        apex = np.array([0.0, 0.0, apex_height], dtype=np.float32)
 
        # Sampling budget
        n_base = int(n_points * 0.35)
        n_lateral = n_points - n_base
 
        pts = []
 
        # Base (fan triangulation)
        for _ in range(n_base):
            i = np.random.randint(n_sides)
            v0 = base_center
            v1 = base_vertices[i]
            v2 = base_vertices[(i + 1) % n_sides]
            r1, r2 = np.random.rand(2).astype(np.float32)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            pts.append(r1 * v0 + r2 * v1 + (1 - r1 - r2) * v2)
 
        # Lateral faces
        for _ in range(n_lateral):
            i = np.random.randint(n_sides)
            v0 = apex
            v1 = base_vertices[i]
            v2 = base_vertices[(i + 1) % n_sides]
            r1, r2 = np.random.rand(2).astype(np.float32)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            pts.append(r1 * v0 + r2 * v1 + (1 - r1 - r2) * v2)
 
        pts = np.array(pts, dtype=np.float32)
        pts = pts - pts.mean(axis=0, keepdims=True)
        return pts
 
    pcs: List[np.ndarray] = []
    labels: List[List[np.ndarray]] = []
    meta_n: List[int] = []
 
    for n_sides in n_sides_list:
        for _ in range(per_class):
            base_pc = generate_ngonal_pyramid(n_sides, num_points)
 
            # Hard case: random SE(3)
            Rw = random_rotation_matrix()
            tw = np.random.uniform(-trans_range, trans_range, size=(3,)).astype(np.float32)
            pc_world = (Rw @ base_pc.T).T + tw
 
            pcs.append(pc_world.astype(np.float32))
            meta_n.append(n_sides)
 
            # GT axis in WORLD coords (local +Z rotated by Rw)
            z_local = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            z_world = (Rw @ z_local.reshape(3, 1)).reshape(3).astype(np.float32)
            z_world = z_world / (np.linalg.norm(z_world) + 1e-8)
 
            # GT angles: all non-trivial rotations around axis
            gt_list = []
            for k in range(1, n_sides):
                ang = (2.0 * np.pi * k) / float(n_sides)
                gt_list.append(np.concatenate([z_world, np.array([ang], dtype=np.float32)]))
            labels.append(gt_list)
 
    return np.array(pcs, dtype=np.float32), labels, meta_n
 
 
@torch.no_grad()
def evaluate_rotation_symmetry_by_correspondence(
    point_clouds: np.ndarray,
    gt_labels: List[List[np.ndarray]],
    inference_model: DCPInference,
    dist_threshold: float = 0.05,
    chamfer_threshold: float = 0.01,
) -> Tuple[float, float, float, float]:
    device = inference_model.device
    precision_sum = 0.0
    recall_sum = 0.0
    low_chamfer_sum = 0.0
    low_chamfer_count = 0
    total_samples = len(point_clouds)
 
    gt_lines: List[str] = []
    out_gt_path = osp.join(osp.dirname(__file__), 'dcp_rotation_symmetry.txt')
 
    pbar = tqdm(range(total_samples), desc='Evaluating samples', dynamic_ncols=True)
    for i in pbar:
        pc = point_clouds[i].astype(np.float32)
 
        gt_params_list = gt_labels[i]
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
 
            angle_diff = torch.abs(pred_angles_expanded - gt_angles_expanded)
            angle_diff = torch.minimum(angle_diff, 2 * np.pi - angle_diff)
 
            angle_weight = 0.1
            metric = axis_dist + angle_weight * angle_diff / np.pi
 
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
 
 
def main():
    # Optional perf toggles (safe)
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
 
    parser = argparse.ArgumentParser(description='DCP Rotation Symmetry Evaluation (GPU RANSAC)')
    default_model = None
    pretrained_dir = osp.join(osp.dirname(__file__), 'pretrained')
    if osp.exists(pretrained_dir):
        if osp.exists(osp.join(pretrained_dir, 'dcp_v2.t7')):
            default_model = osp.join(pretrained_dir, 'dcp_v2.t7')
        elif osp.exists(osp.join(pretrained_dir, 'dcp_v1.t7')):
            default_model = osp.join(pretrained_dir, 'dcp_v1.t7')
 
    parser.add_argument('--model_path', type=str, default=default_model,
                        help=f'Path to DCP model weights (default: {default_model if default_model else "required"})')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points per point cloud')
    parser.add_argument('--dist_threshold', type=float, default=0.1, help='Distance threshold for matching')
    parser.add_argument('--chamfer_threshold', type=float, default=0.005, help='Chamfer distance threshold')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Embedding network type')
    parser.add_argument('--pointer', type=str, default='transformer', choices=['identity', 'transformer'], help='Pointer type')
    parser.add_argument('--head', type=str, default='svd', choices=['mlp', 'svd'], help='Head type')
    parser.add_argument('--emb_dims', type=int, default=512, help='Embedding dimensions')
 
    # Dataset params
    parser.add_argument('--per_class', type=int, default=50, help='Samples per shape (default 50 -> total 200)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset generation')
    parser.add_argument('--apex_height', type=float, default=1.5, help='Apex height of pyramids')
    parser.add_argument('--trans_range', type=float, default=0.5, help='Translation range for random SE(3)')
 
    args = parser.parse_args()
 
    if args.model_path is None:
        parser.error("--model_path is required. Available models: pretrained/dcp_v1.t7, pretrained/dcp_v2.t7")
 
    infer = DCPInference(
        args.model_path,
        device=args.device,
        emb_nn=args.emb_nn,
        pointer=args.pointer,
        head=args.head,
        emb_dims=args.emb_dims
    )
 
    print(f"Generating toy dataset: n-gonal pyramids (3/4/5/6) x {args.per_class} each "
          f"= {4*args.per_class} samples, {args.num_points} points each...")
    pcs, labels, meta_n = generate_toy_dataset(
        per_class=args.per_class,
        num_points=args.num_points,
        seed=args.seed,
        n_sides_list=[3, 4, 5, 6],
        apex_height=args.apex_height,
        trans_range=args.trans_range,
    )
    print(f"Generated {len(pcs)} point clouds. Class counts: "
          f"3:{sum(n==3 for n in meta_n)}, 4:{sum(n==4 for n in meta_n)}, "
          f"5:{sum(n==5 for n in meta_n)}, 6:{sum(n==6 for n in meta_n)}")
 
    # Save one sample
    sample_path = osp.join(osp.dirname(__file__), f'pyramid_n{meta_n[0]}_sample.ply')
    with open(sample_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pcs[0].shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pcs[0]:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"Saved sample for viewing: {sample_path}")
 
    precision, recall, f1, avg_ch = evaluate_rotation_symmetry_by_correspondence(
        pcs, labels, infer,
        dist_threshold=args.dist_threshold,
        chamfer_threshold=args.chamfer_threshold
    )
 
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Chamfer:   {avg_ch:.6f}")
 
 
if __name__ == '__main__':
    main()