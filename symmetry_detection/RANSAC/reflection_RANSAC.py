from typing import Tuple

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance


@torch.no_grad()
def _planes_from_pairs(
    ref_corr: np.ndarray,
    src_corr: np.ndarray,
    reflection_matrix: np.ndarray,
    point_cloud: torch.Tensor,
    device: torch.device,
    inlier_dist_threshold: float = 0.01,
    num_iterations: int = 100,
    count_entire: bool = False,
    refinement_criteria: float = 10.0,
    available_inlier_threshold: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    RANSAC-style plane extraction from correspondence pairs.
    Returns arrays of plane normals, offsets, centers, and chamfer distances.
    """
    if ref_corr.shape[0] < 3 or src_corr.shape[0] < 3:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    M_src = (reflection_matrix.astype(np.float32) @ src_corr.T).T

    diff = ref_corr - M_src
    norms = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
    n_per_pair = diff / norms
    c_per_pair = 0.5 * (ref_corr + M_src)
    d_per_pair = np.sum(n_per_pair * c_per_pair, axis=1)

    n_pairs = n_per_pair.shape[0]
    n_batch = torch.from_numpy(n_per_pair).to(device, dtype=torch.float32)
    d_batch = torch.from_numpy(d_per_pair).to(device, dtype=torch.float32)

    ref_corr_torch = torch.from_numpy(ref_corr).to(device, dtype=torch.float32)
    M_src_torch = torch.from_numpy(M_src).to(device, dtype=torch.float32)

    inlier_matrix = torch.zeros((n_pairs, n_pairs), dtype=torch.bool, device=device)
    batch_size = 128
    if count_entire:
        entire_inlier_matrix = torch.zeros((n_pairs), dtype=torch.int64, device=device)

    for start_idx in range(0, n_pairs, batch_size):
        end_idx = min(start_idx + batch_size, n_pairs)

        n_chunk = n_batch[start_idx:end_idx]
        d_chunk = d_batch[start_idx:end_idx]

        H_chunk = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0) - 2.0 * torch.bmm(
            n_chunk.unsqueeze(2), n_chunk.unsqueeze(1)
        )

        shifts = d_chunk.unsqueeze(1) * n_chunk
        ref_shifted = ref_corr_torch.unsqueeze(0) - shifts.unsqueeze(1)
        M_src_shifted = M_src_torch.unsqueeze(0) - shifts.unsqueeze(1)

        ref_reflected = torch.bmm(ref_shifted, H_chunk)
        M_src_reflected = torch.bmm(M_src_shifted, H_chunk)

        dist1 = torch.norm(ref_reflected - M_src_shifted, dim=2)
        dist2 = torch.norm(M_src_reflected - ref_shifted, dim=2)
        avg_dist = (dist1 + dist2) / 2.0

        inlier_matrix[start_idx:end_idx, :] = avg_dist < inlier_dist_threshold

        if count_entire:
            pcd_shifted = point_cloud.unsqueeze(0) - shifts.unsqueeze(1)
            pcd_reflected = torch.bmm(pcd_shifted, H_chunk)
            pcd_inlier_matrix = torch.cdist(pcd_reflected, pcd_shifted, p=2) < inlier_dist_threshold
            entire_inlier_matrix[start_idx:end_idx] = pcd_inlier_matrix.sum(dim=(1, 2))

    available_mask = np.ones(n_pairs, dtype=bool)
    result_normals, result_ds, result_centers = [], [], []

    # if refinement_criteria != 10.0:
    #     criteria_threshold = n_pairs * refinement_criteria / 100.0
    # else:
    #     criteria_threshold = 10
    criteria_threshold = n_pairs * refinement_criteria / 100.0

    for _ in range(num_iterations):
        # print(f"available pairs: {available_mask.sum()} / {n_pairs}")
        if not available_mask.any() or available_mask.sum() < criteria_threshold:
            break

        if count_entire:
            best_idx = np.argmax(entire_inlier_matrix.cpu().numpy() * available_mask).item()
            available_inliers = inlier_matrix[:, available_mask].sum(dim=1).cpu().numpy()
            if available_inliers[best_idx] < available_inlier_threshold:
                available_mask[inlier_matrix[best_idx, :].cpu().numpy()] = False
                continue
        else:
            available_inliers = inlier_matrix[:, available_mask].sum(dim=1).cpu().numpy()
            best_idx = np.argmax(available_inliers)
            if available_inliers[best_idx] < criteria_threshold:
                break

        inlier_local = inlier_matrix[best_idx, :].cpu().numpy() & available_mask

        X_inliers = M_src[inlier_local]
        Y_inliers = ref_corr[inlier_local]

        diffs = Y_inliers - X_inliers
        diffs_centered = diffs - diffs.mean(axis=0)
        cov = diffs_centered.T @ diffs_centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_dir = eigvecs[:, np.argmax(eigvals)]
        n_vec = principal_dir / (np.linalg.norm(principal_dir) + 1e-8)

        midpoints = c_per_pair[inlier_local]
        mean_mid = midpoints.mean(axis=0)
        d_val = -float(n_vec.dot(mean_mid))

        result_normals.append(n_vec.astype(np.float32))
        result_ds.append(np.float32(d_val))
        result_centers.append(mean_mid.astype(np.float32))

        available_mask[inlier_local] = False

    if len(result_normals) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    result_normals_np = np.array(result_normals, dtype=np.float32)
    result_ds_np = np.array(result_ds, dtype=np.float32)
    result_centers_np = np.array(result_centers, dtype=np.float32)

    n_final = len(result_normals)
    chamfer_values = np.zeros(n_final, dtype=np.float32)
    I = torch.eye(3, device=device, dtype=torch.float32)

    for i in range(n_final):
        n_vec = torch.from_numpy(result_normals_np[i]).to(device)
        d_val = torch.tensor(result_ds_np[i], device=device)

        shift = d_val * n_vec
        X_shifted = point_cloud - shift.unsqueeze(0)
        H = I - 2.0 * torch.outer(n_vec, n_vec)
        X_reflected = X_shifted @ H

        chamfer_val, _ = chamfer_distance(
            X_shifted.unsqueeze(0), X_reflected.unsqueeze(0)
        )
        chamfer_values[i] = chamfer_val.item()

    return (
        result_normals_np,
        result_ds_np,
        result_centers_np,
        chamfer_values,
    )
