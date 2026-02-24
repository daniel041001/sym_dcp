import os.path as osp
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# try:
#     from .RANSAC.rotation_RANSAC import _rotations_from_pairs_gpu
# except ImportError:
#     from symmetry_detection.RANSAC.rotation_RANSAC import _rotations_from_pairs_gpu

from .RANSAC.rotation_RANSAC_JH import _rotations_from_pairs_gpu


def get_correspondences(
    inference_model: None,
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
    inference_model: None,
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
