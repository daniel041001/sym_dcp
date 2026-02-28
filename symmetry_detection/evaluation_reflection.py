import os.path as osp
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from .RANSAC.reflection_RANSAC import _planes_from_pairs
except ImportError:
    from symmetry_detection.RANSAC.reflection_RANSAC import _planes_from_pairs


def get_correspondences(
    inference_model: None,
    point_cloud: np.ndarray,
    reflection_matrix: np.ndarray,
    return_scores: bool = False
) -> tuple:
    """
    Get correspondences between a point cloud and its reflection.
    GeoTransformer의 get_correspondences와 동일한 인터페이스.
    
    Args:
        inference_model: DCPInference instance
        point_cloud: Input point cloud (N, 3)
        reflection_matrix: Reflection matrix (3, 3)
        return_scores: If True, also return correspondence scores
    
    Returns:
        base: Base point cloud (N, 3)
        reflected: Reflected point cloud (N, 3)
        base_corr: Correspondence points from base (K, 3)
        reflected_corr: Correspondence points from reflected (K, 3)
        scores: (Optional) Correspondence scores matrix
    """
    base = point_cloud.astype(np.float32)
    reflected = (reflection_matrix.astype(np.float32) @ base.T).T
    
    if return_scores:
        base_corr, reflected_corr, scores = inference_model.get_correspondences(
            base, reflected, return_scores=True
        )
        return base, reflected, base_corr, reflected_corr, scores
    else:
        base_corr, reflected_corr = inference_model.get_correspondences(
            base, reflected, return_scores=False
        )
        return base, reflected, base_corr, reflected_corr
    
@torch.no_grad()
def evaluate_reflection_symmetry_by_correspondence(
    point_clouds: np.ndarray,
    gt_labels: List[List[np.ndarray]],
    reflection_matrices: List[np.ndarray],
    inference_model: None,
    dist_threshold: float = 0.05,
    chamfer_threshold: float = 1e-3,
    count_entire: bool = False,
    refinement_criteria: float = 10.0,
    available_inlier_threshold: int = 10,
) -> Tuple[float, float, float, float]:
    """
    Evaluate reflection symmetry detection using DCP correspondences.
    Same interface as GeoTransformer's evaluate_reflection_symmetry_by_correspondence.
    """
    device = inference_model.device
    precision_sum = 0.0
    recall_sum = 0.0
    low_chamfer_sum = 0.0  # accumulate chamfer for cd <= chamfer_threshold
    low_chamfer_count = 0
    total_samples = len(point_clouds)
    total_pred = 0  # kept for compatibility in progress display if needed

    I = torch.eye(3, device=device, dtype=torch.float32)

    # logging: save idx, normal, min distance to GT (sign-invariant)
    gt_lines: List[str] = []
    out_gt_path = osp.join(osp.dirname(__file__), 'dcp_cluster_planes_inlier_xy_8.txt')

    pbar = tqdm(range(total_samples), desc='Evaluating samples', dynamic_ncols=True)
    for i in pbar:
        pc = point_clouds[i].astype(np.float32)
        
        # No PCA alignment - use original point cloud
        # Extract GT labels directly (no rotation needed)
        gt_params_list = gt_labels[i]
        gt_normals_list = [p[:3] for p in gt_params_list]
        gt_offsets_list = [float(p[3]) for p in gt_params_list]
        
        gt_normals_tensor = torch.stack([torch.tensor(n, dtype=torch.float32, device=device) for n in gt_normals_list])
        gt_normals_tensor = gt_normals_tensor / (torch.norm(gt_normals_tensor, dim=1, keepdim=True) + 1e-8)
        gt_offsets_tensor = torch.tensor(gt_offsets_list, dtype=torch.float32, device=device)  # (num_gt,)

        pred_normals_torch: List[torch.Tensor] = []
        pred_chamfers: List[float] = []
        pred_ds: List[float] = []
        # number of planes kept after thresholding (set later)
        planes_kept = 0

        # Use only XY-plane reflection
        M_xy = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)  # reflect across XY (z -> -z)

        # Use original point cloud directly
        base = pc.astype(np.float32)

        # Get correspondences from XY reflection only
        _, reflected, ref_corr_all, src_corr_all = get_correspondences(
            inference_model, base, M_xy
        )
        
        num_correspondences = ref_corr_all.shape[0]
        
        # DEBUG: Check correspondence quality before transformation
        if ref_corr_all.shape[0] > 0 and src_corr_all.shape[0] > 0:
            # src_corr_all is in reflected space, we need to transform it back
            # But first, let's understand what we have:
            # - ref_corr_all: points from base point cloud
            # - src_corr_all: points from reflected point cloud (in reflected space)
            
            # Transform src_corr back to original space
            src_corr_all = (M_xy.astype(np.float32) @ src_corr_all.T).T
            
            # DEBUG: Verify the transformation
            # After transformation, src_corr_all should be close to ref_corr_all for good correspondences
        else:
            ref_corr_all = np.zeros((0, 3), dtype=np.float32)
            src_corr_all = np.zeros((0, 3), dtype=np.float32)
        
        # Count correspondence pairs with distance < threshold
        if ref_corr_all.shape[0] > 0 and src_corr_all.shape[0] > 0:
            pair_distances = np.linalg.norm(ref_corr_all - src_corr_all, axis=1)
            num_close_pairs = np.sum(pair_distances < 0.01)  # Increased threshold for debugging
            avg_pair_distance = np.mean(pair_distances) if len(pair_distances) > 0 else 0.0
        else:
            num_close_pairs = 0
            avg_pair_distance = 0.0
        
        # Centerize point cloud at origin (do this once at the beginning)
        X = torch.from_numpy(base).float().to(device)
        X_center = X.mean(dim=0)
        X_orig = X - X_center  # Center at origin
        
        # Adjust GT offsets for centerizing: d_adjusted = d - n·X_center
        gt_offsets_adjusted = gt_offsets_tensor - (gt_normals_tensor @ X_center)  # (num_gt,)

        # Get planes from ALL correspondence pairs (RANSAC-style with inlier counting)
        # Use identity matrix since src_corr_all is already transformed to original space
        I_matrix = np.eye(3, dtype=np.float32)
        n_np, d_np, _, chamfer_np = _planes_from_pairs(ref_corr_all, src_corr_all, I_matrix, X_orig, device, 
                                                        inlier_dist_threshold=0.007, num_iterations=100,
                                                        count_entire=count_entire, refinement_criteria=refinement_criteria,
                                                        available_inlier_threshold=available_inlier_threshold)

        # Process planes if any found
        if n_np.shape[0] > 0:
            # Convert to torch tensors
            n_batch = torch.from_numpy(n_np).to(device)  # (K, 3)
            d_batch = torch.from_numpy(d_np).to(device, dtype=torch.float32)  # (K,)
            chamfer_batch = torch.from_numpy(chamfer_np).to(device, dtype=torch.float32)  # (K,)
            
            # Adjust d by the shift amount for all planes at once
            d_adjusted_batch = d_batch - (n_batch @ X_center)  # (K,)
            
            # Filter planes by chamfer distance threshold - vectorized
            filter_mask = chamfer_batch < chamfer_threshold
            
            if filter_mask.any():
                # Apply filter (keep on GPU for efficiency)
                pred_normals_torch = n_batch[filter_mask].unbind(0)
                pred_ds = d_adjusted_batch[filter_mask].tolist()
                pred_chamfers = chamfer_batch[filter_mask].tolist()
            else:
                pred_normals_torch = []
                pred_ds = []
                pred_chamfers = []

        if len(pred_normals_torch) == 0:
            precision_sum += 1  # same handling as inference.py
        else:
            # Keep data on GPU for efficiency (avoid unnecessary conversions)
            keep_normals = torch.stack(pred_normals_torch)  # (N,3)
            keep_ds = torch.tensor(pred_ds, dtype=torch.float32, device=device)  # (N,)
            keep_chamfers = torch.tensor(pred_chamfers, dtype=torch.float32, device=device)  # (N,)
            planes_kept = keep_normals.shape[0]

            # Track low-chamfer stats for reporting (all are already below threshold)
            low_chamfer_sum += float(keep_chamfers.sum().item())
            low_chamfer_count += len(keep_chamfers)

            # Compute GT distance metrics once for all planes (vectorized)
            # Expand dimensions for broadcasting: (num_pred, 1, 3) and (1, num_gt, 3)
            pred_expanded = keep_normals.unsqueeze(1)  # (num_pred, 1, 3)
            gt_expanded = gt_normals_tensor.unsqueeze(0)  # (1, num_gt, 3)
            
            # Compute distances: (num_pred, num_gt)
            dist_pos = torch.norm(pred_expanded - gt_expanded, dim=2)
            dist_neg = torch.norm(pred_expanded + gt_expanded, dim=2)
            min_norm = torch.minimum(dist_pos, dist_neg)
            
            # Determine which direction is closer (same or opposite)
            same_direction = dist_pos < dist_neg  # (num_pred, num_gt)
            
            # Compute offset distances based on direction
            # If same direction: |d_pred - d_gt|
            # If opposite direction: |d_pred + d_gt|
            pred_d_expanded = keep_ds.unsqueeze(1)  # (num_pred, 1)
            gt_d_expanded = gt_offsets_adjusted.unsqueeze(0)  # (1, num_gt)
            
            offset_dist_same = torch.abs(pred_d_expanded - gt_d_expanded)  # (num_pred, num_gt)
            offset_dist_opp = torch.abs(pred_d_expanded + gt_d_expanded)  # (num_pred, num_gt)
            
            # Select appropriate offset distance based on normal direction
            offset_dist = torch.where(same_direction, offset_dist_same, offset_dist_opp)
            
            # Combine normal and offset distances: (num_pred, num_gt)
            metric = torch.sqrt(min_norm * min_norm + offset_dist * offset_dist)
            
            precision_sum_per_threshold = 0.0
            recall_sum_per_threshold = 0.0
            for mutli_threshold in range(1, 5):
                # Match matrix: (num_pred, num_gt)
                match_matrix = metric < dist_threshold * mutli_threshold  # Adjust threshold for different levels of strictness
                
                # Precision: TP / #predicted
                # TP = number of predicted planes that match at least one GT
                matches_per_pred = match_matrix.any(dim=1)  # (num_pred,)
                tp_precision = int(matches_per_pred.sum().item())
                precision_sum_per_threshold += tp_precision / keep_normals.shape[0]
                
                # Recall: TP / #GT
                # TP = number of GT planes that are matched by at least one prediction
                matches_per_gt = match_matrix.any(dim=0)  # (num_gt,)
                tp_recall = int(matches_per_gt.sum().item())
                recall_sum_per_threshold += tp_recall / len(gt_normals_tensor)
            precision_sum += precision_sum_per_threshold / 4.0
            recall_sum += recall_sum_per_threshold / 4.0
            
            # Logging: compute min GT distance for each pred plane
            min_gt_dist_per_pred = metric.min(dim=1)[0]  # (num_pred,)
            
            # Vectorized logging - all keep_normals are already below chamfer_threshold
            for k in range(keep_normals.shape[0]):
                n = keep_normals[k]
                d_adjusted = keep_ds[k]
                chamfer_val = keep_chamfers[k].item()
                min_metric = min_gt_dist_per_pred[k].item()
                gt_lines.append(
                    f"{i}\t{float(n[0]):.6f}\t{float(n[1]):.6f}\t{float(n[2]):.6f}\t{float(d_adjusted):.6f}\t{chamfer_val:.6f}\t{min_metric:.6f}"
                )

            total_pred += len(pred_chamfers)

        # update progress postfix
        done = i + 1
        pbar.set_postfix(
            corr=num_correspondences,
            close=num_close_pairs,
            avg_dist=f"{avg_pair_distance:.4f}" if ref_corr_all.shape[0] > 0 else "0.0",
            planes=planes_kept,
            P=(precision_sum / done) if done > 0 else 0.0,
            R=(recall_sum / done) if done > 0 else 0.0,
        )

    # write log file
    try:
        with open(out_gt_path, 'w') as f:
            f.write("idx\tnx\tny\tnz\td\tchamfer\tmin_gt_dist\n")
            for line in gt_lines:
                f.write(line + "\n")
    except Exception:
        pass

    precision = precision_sum / total_samples if total_samples > 0 else 0.0
    recall = recall_sum / total_samples if total_samples > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_chamfer = low_chamfer_sum / low_chamfer_count if low_chamfer_count > 0 else 0.0
    return precision, recall, f1, avg_chamfer
