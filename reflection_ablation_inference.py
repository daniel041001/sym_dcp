#!/usr/bin/env python3
"""
MSR (Mirror Symmetry via Registration) for 3D point clouds
Based on:
Cicconet et al., "Finding Mirror Symmetry via Registration", ICCV 2017 Workshop.
 
Pipeline (paper Sec 3.1):
1) Choose initial reflection plane (point p, unit normal v).
2) Reflect points: x -> S_v x + 2 d v, where d = <p, v>, S_v = I - 2 v v^T.
3) Register reflected set Q to original P via rigid transform x -> R0 x + t (ICP).
4) Symmetry plane normal v_bar = eigenvector of (S_v R0^T) with eigenvalue -1.
5) Point on plane p_bar = 1/2 (R0 (2 d v) + t).
"""
 
from __future__ import annotations
import argparse
import math
import os
import os.path as osp
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List
 
import numpy as np
 
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x
 
try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception as e:
    raise ImportError("This script needs scipy (for cKDTree). Install: pip install scipy") from e
 
 
@dataclass
class Plane:
    point: np.ndarray   # (3,)
    normal: np.ndarray  # (3,), unit
    # Hessian normal form: n·x + offset = 0  (offset = -n·p)
    offset: float
    # ICP quality (lower is better)
    rmse: float
    # which init normal used
    init_normal: np.ndarray
 
 
def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n
 
 
def reflect_points(points: np.ndarray, p: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Reflect 3D points about plane H = {x : <x - p, v> = 0}, with ||v||=1.
 
    Paper notation:
      S_v = I - 2 v v^T
      d = <p, v>
      x -> S_v x + 2 d v
    """
    v = _normalize(v.reshape(3))
    p = p.reshape(3)
    I = np.eye(3, dtype=np.float64)
    Sv = I - 2.0 * np.outer(v, v)
    d = float(np.dot(p, v))
    # points are rows; apply column-form x' = Sv x + 2 d v => row form: x' = x Sv^T + 2 d v
    reflected = points @ Sv.T + (2.0 * d) * v
    return reflected, d, Sv
 
 
def best_fit_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rigid transform (R, t) that maps src -> dst in least squares sense.
    Column-vector convention: x' = R x + t.
    Inputs are Nx3 rows, but math is the same.
    """
    if src.shape != dst.shape:
        raise ValueError("src and dst must have same shape (Nx3).")
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 points for a stable rigid transform.")
 
    src_cent = src.mean(axis=0)
    dst_cent = dst.mean(axis=0)
    X = src - src_cent
    Y = dst - dst_cent
 
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # fix reflection in rotation (ensure det(R)=+1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
 
    t = dst_cent - (R @ src_cent)
    return R, t
 
 
def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: Optional[float] = None,
    rejection_quantile: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simple point-to-point ICP registering 'source' to 'target'.
    Returns (R, t, rmse) with x_target ≈ R x_source + t.
 
    Options:
      - max_correspondence_distance: discard matches farther than this distance.
      - rejection_quantile: keep only the closest fraction of correspondences (e.g., 0.9 keeps 90% best).
    """
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source must be Nx3.")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target must be Mx3.")
 
    tree = cKDTree(target)
 
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
 
    prev_rmse = np.inf
 
    for _ in range(max_iterations):
        src_trans = source @ R_total.T + t_total  # row form
 
        dists, idx = tree.query(src_trans, k=1, workers=-1)
        dists = dists.astype(np.float64)
 
        # distance-based filtering
        keep = np.ones_like(dists, dtype=bool)
        if max_correspondence_distance is not None:
            keep &= (dists <= float(max_correspondence_distance))
 
        if rejection_quantile < 1.0:
            q = float(np.quantile(dists[keep], rejection_quantile)) if np.any(keep) else np.inf
            keep &= (dists <= q)
 
        if np.count_nonzero(keep) < 3:
            # not enough correspondences; stop early
            break
 
        src_kept = src_trans[keep]
        dst_kept = target[idx[keep]]
 
        R_delta, t_delta = best_fit_transform(src_kept, dst_kept)
 
        # compose: new_total(x) = R_delta (R_total x + t_total) + t_delta
        R_total = R_delta @ R_total
        t_total = (R_delta @ t_total) + t_delta
 
        rmse = float(np.sqrt(np.mean(np.square(dists[keep]))))
        if abs(prev_rmse - rmse) < tolerance:
            prev_rmse = rmse
            break
        prev_rmse = rmse
 
    return R_total, t_total, prev_rmse
 
 
def msr_symmetry_plane_single_init(
    P: np.ndarray,
    init_normal: np.ndarray,
    icp_iters: int = 50,
    icp_tol: float = 1e-6,
    max_corr_dist: Optional[float] = None,
    rejection_quantile: float = 0.95,
) -> Plane:
    """
    Run MSR once for a given initial normal (paper suggests trying multiple).
    Uses p as center of mass (mean).
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P must be Nx3.")
 
    p0 = P.mean(axis=0)
    v0 = _normalize(np.asarray(init_normal, dtype=np.float64))
 
    # Step 2: reflect to get Q
    Q, d, Sv = reflect_points(P, p0, v0)
 
    # Step 3: register Q to P (rigid): x -> R0 x + t
    R0, t, rmse = icp_point_to_point(
        source=Q,
        target=P,
        max_iterations=icp_iters,
        tolerance=icp_tol,
        max_correspondence_distance=max_corr_dist,
        rejection_quantile=rejection_quantile,
    )
 
    # Step 4: v_bar = eigenvector of Sv R0^T with eigenvalue -1
    M = Sv @ R0.T
    eigvals, eigvecs = np.linalg.eig(M)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
 
    k = int(np.argmin(np.abs(eigvals + 1.0)))
    v_bar = _normalize(eigvecs[:, k])
 
    # Step 5: point on plane p_bar = 1/2 ( R0(2 d v) + t )
    p_bar = 0.5 * (R0 @ ((2.0 * d) * v0) + t)
 
    offset = float(-np.dot(v_bar, p_bar))  # n·x + offset = 0
 
    return Plane(
        point=p_bar,
        normal=v_bar,
        offset=offset,
        rmse=rmse,
        init_normal=v0,
    )
 
 
def msr_symmetry_plane(
    P: np.ndarray,
    init_normals: Iterable[np.ndarray] = (np.array([1.0, 0.0, 0.0]),
                                         np.array([0.0, 1.0, 0.0]),
                                         np.array([0.0, 0.0, 1.0])),
    icp_iters: int = 50,
    icp_tol: float = 1e-6,
    max_corr_dist: Optional[float] = None,
    rejection_quantile: float = 0.95,
) -> Plane:
    """
    Try multiple initial planes (paper 3D experiments try 3 orthogonal normals and pick best registration).
    Picks the plane with the lowest ICP RMSE.
    """
    best: Optional[Plane] = None
    for n0 in init_normals:
        plane = msr_symmetry_plane_single_init(
            P,
            init_normal=np.asarray(n0, dtype=np.float64),
            icp_iters=icp_iters,
            icp_tol=icp_tol,
            max_corr_dist=max_corr_dist,
            rejection_quantile=rejection_quantile,
        )
        if best is None or plane.rmse < best.rmse:
            best = plane
 
    assert best is not None
    return best
 
 
def load_xyz(path: str) -> np.ndarray:
    """
    Load a simple whitespace-separated Nx3 text file.
    (For .ply/.pcd, use Open3D in your own pipeline and pass the Nx3 numpy array to msr_symmetry_plane.)
    """
    pts = np.loadtxt(path, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] < 3:
        raise ValueError("File must contain at least 3 columns (x y z).")
    return pts[:, :3]
 
 
def maybe_subsample(P: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if P.shape[0] <= max_points:
        return P
    rng = np.random.default_rng(seed)
    idx = rng.choice(P.shape[0], size=max_points, replace=False)
    return P[idx]
 
 
def load_preprocessed_dataset(dataset_root: str) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """
    Load preprocessed reflection-symmetry dataset.
    동일한 포맷의 데이터셋을 dcp_inference_1.py와 같이 사용하기 위함.
    """
    import pickle
    import glob
 
    pcd_file = osp.join(dataset_root, "point_clouds.pkl")
    labels_file = osp.join(dataset_root, "labels.pkl")
 
    if osp.exists(pcd_file) and osp.exists(labels_file):
        with open(pcd_file, "rb") as f:
            point_clouds = pickle.load(f)
        with open(labels_file, "rb") as f:
            labels = pickle.load(f)
    else:
        pcd_files = sorted(glob.glob(osp.join(dataset_root, "*.npy")))
        if len(pcd_files) == 0:
            raise FileNotFoundError(f"No data found in {dataset_root}")
 
        point_clouds_list: List[np.ndarray] = []
        labels: List[List[np.ndarray]] = []
 
        for pf in pcd_files:
            data = np.load(pf, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                item = data.item()
                point_clouds_list.append(item["points"])
                labels.append(item.get("labels", []))
            else:
                point_clouds_list.append(data)
                label_file = pf.replace(".npy", "_labels.npy")
                if osp.exists(label_file):
                    label_data = np.load(label_file, allow_pickle=True)
                    labels.append(label_data.tolist() if hasattr(label_data, "tolist") else [])
                else:
                    labels.append([])
 
        point_clouds = np.array(point_clouds_list)
 
    return point_clouds, labels
 
 
def evaluate_msr_reflection_symmetry(
    point_clouds: np.ndarray,
    gt_labels: List[List[np.ndarray]],
    dist_threshold: float = 0.05,
    max_points: int = 20000,
    seed: int = 0,
    icp_iters: int = 60,
    icp_tol: float = 1e-6,
    max_corr_dist: Optional[float] = None,
    rejection_quantile: float = 0.95,
) -> Tuple[float, float, float]:
    """
    MSR로 추정한 반사 대칭 평면을 이용해 F1 score를 계산하는 평가 루프.
    - 각 포인트 클라우드마다 MSR로 1개의 평면을 예측.
    - GT 평면 파라미터([nx, ny, nz, d], n·x - d = 0 가정)와의 거리로
      precision / recall / F1을 계산.
    """
    num_samples = len(point_clouds)
    if num_samples == 0:
        return 0.0, 0.0, 0.0
 
    precision_sum = 0.0
    recall_sum = 0.0
 
    pred_labels: List[np.ndarray] = []
    for i in tqdm(range(num_samples), desc="Evaluating MSR (ICP)", dynamic_ncols=True):
        P = np.asarray(point_clouds[i], dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("Each point cloud must be of shape (N, 3).")
 
        # 서브샘플링 (선택사항, dcp_inference_1.py 의 max_points와 동일한 역할)
        P_sub = maybe_subsample(P, max_points=max_points, seed=seed)
 
        # MSR로 평면 하나 예측
        plane = msr_symmetry_plane(
            P_sub,
            icp_iters=icp_iters,
            icp_tol=icp_tol,
            max_corr_dist=max_corr_dist,
            rejection_quantile=rejection_quantile,
        )
 
        # 예측 평면 파라미터: n (unit) 과 d = n·p  (식: n·x - d = 0 가정)
        n_pred = plane.normal.astype(np.float64)
        n_pred /= np.linalg.norm(n_pred) + 1e-12
        d_pred = float(np.dot(n_pred, plane.point.astype(np.float64)))

        # for visualization
        pred_labels.append(np.hstack([n_pred.reshape(1, 3), np.array([[d_pred]], dtype=np.float64)]))
 
        # GT 평면들
        gt_params = gt_labels[i]
        if len(gt_params) == 0:
            # GT 가 없으면 precision 은 1, recall 은 0 으로 처리
            precision_sum += 1.0
            continue
 
        gt_normals = []
        gt_ds = []
        for p in gt_params:
            p = np.asarray(p, dtype=np.float64)
            if p.shape[0] < 4:
                continue
            n_gt = p[:3]
            if np.linalg.norm(n_gt) < 1e-12:
                continue
            n_gt = n_gt / (np.linalg.norm(n_gt) + 1e-12)
            d_gt = float(p[3])
            gt_normals.append(n_gt)
            gt_ds.append(d_gt)
 
        if len(gt_normals) == 0:
            precision_sum += 1.0
            continue
 
        gt_normals_arr = np.stack(gt_normals, axis=0)  # (G, 3)
        gt_ds_arr = np.asarray(gt_ds, dtype=np.float64)  # (G,)
 
        # 단 하나의 예측 평면만 있으므로 간단히 계산
        # 방향 부호까지 고려해서 (n,d) 와 (±n_gt, ±d_gt) 거리 중 작은 값을 사용
        # normal distance
        diff_pos_n = np.linalg.norm(n_pred[None, :] - gt_normals_arr, axis=1)
        diff_neg_n = np.linalg.norm(n_pred[None, :] + gt_normals_arr, axis=1)
        same_dir = diff_pos_n <= diff_neg_n
        min_norm = np.minimum(diff_pos_n, diff_neg_n)
 
        # offset distance
        d_pred_arr = np.full_like(gt_ds_arr, d_pred, dtype=np.float64)
        off_same = np.abs(d_pred_arr - gt_ds_arr)
        off_opp = np.abs(d_pred_arr + gt_ds_arr)
        offset_dist = np.where(same_dir, off_same, off_opp)
 
        metric = np.sqrt(min_norm * min_norm + offset_dist * offset_dist)
 
        # threshold 이하인 매칭이 있는지 확인
        matched_gt = metric < dist_threshold
        tp_precision = 1 if np.any(matched_gt) else 0
        precision_sum += tp_precision  # 예측은 항상 1개이므로 /1 생략
 
        tp_recall = 1 if np.any(matched_gt) else 0
        recall_sum += tp_recall / len(gt_normals_arr)
 
    precision = precision_sum / num_samples
    recall = recall_sum / num_samples
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, pred_labels
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Point cloud as Nx3 text (xyz).", nargs="?")
    ap.add_argument("--max_points", type=int, default=20000, help="Randomly subsample to this many points.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--icp_iters", type=int, default=60)
    ap.add_argument("--icp_tol", type=float, default=1e-6)
    ap.add_argument("--max_corr_dist", type=float, default=0.0,
                    help="Max correspondence distance for ICP (0 => disabled).")
    ap.add_argument("--trim", type=float, default=0.95,
                    help="Keep closest fraction of correspondences each ICP iter (robustness).")
    ap.add_argument("--data_root", type=str, default=None,
                    help="If set, evaluate MSR reflection symmetry on preprocessed dataset at this path.")
    ap.add_argument("--dist_threshold", type=float, default=0.05,
                    help="Distance threshold for plane matching when computing F1 score.")
    ap.add_argument("--visualization", action="store_true", help="Save visualization HTML files in vis_qual/ablation/")
    args = ap.parse_args()
 
    max_corr = None if args.max_corr_dist <= 0 else float(args.max_corr_dist)
 
    # Dataset evaluation 모드
    if args.data_root is not None:
        pcs, labels = load_preprocessed_dataset(args.data_root)
        precision, recall, f1, pred_labels = evaluate_msr_reflection_symmetry(
            pcs,
            labels,
            dist_threshold=float(args.dist_threshold),
            max_points=args.max_points,
            seed=args.seed,
            icp_iters=args.icp_iters,
            icp_tol=args.icp_tol,
            max_corr_dist=max_corr,
            rejection_quantile=float(args.trim),
        )
        if args.visualization:
            from vis_qual.visualization_html_reflection import save_visualization
            save_visualization(pcs, labels, save_dir=osp.join("vis_qual", "ablation"), prefix="gt")
            save_visualization(pcs, pred_labels, save_dir=osp.join("vis_qual", "ablation"), prefix="pred")
        print("=== MSR reflection symmetry evaluation (dataset) ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")
        return
 
    # 단일 포인트 클라우드 입력 모드 (기존 기능 유지)
    if not args.input:
        raise SystemExit("Either an input file must be given or --data_root must be set.")
 
    P = load_xyz(args.input)
    P = maybe_subsample(P, args.max_points, seed=args.seed)
 
    plane = msr_symmetry_plane(
        P,
        init_normals=(np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0]),
                      np.array([0.0, 0.0, 1.0])),
        icp_iters=args.icp_iters,
        icp_tol=args.icp_tol,
        max_corr_dist=max_corr,
        rejection_quantile=float(args.trim),
    )
 
    print("=== MSR symmetry plane (point-normal form) ===")
    print(f"point  : {plane.point}")
    print(f"normal : {plane.normal}   (unit)")
    print("Hessian form: n·x + offset = 0")
    print(f"offset : {plane.offset}")
    print(f"ICP RMSE: {plane.rmse}")
    print(f"chosen init normal: {plane.init_normal}")
 
 
if __name__ == "__main__":
    main()