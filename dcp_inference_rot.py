"""
DCP 모델을 사용하여 correspondence를 추출하는 inference 함수
Rotation symmetry detection을 위한 토이 데이터셋 사용
"""
import os
import os.path as osp
import argparse
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import numpy as np
from pytorch3d.loss import chamfer_distance
from model import DCP

# tqdm (with safe fallback)
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


class DCPInference:
    """
    DCP 모델을 사용한 inference wrapper
    Correspondence를 추출할 수 있도록 확장
    """
    
    def __init__(self, model_path: str, device: str = 'cuda:0', args=None, 
                 emb_nn: str = 'dgcnn', pointer: str = 'transformer', 
                 head: str = 'svd', emb_dims: int = 512):
        """
        Initialize DCP inference model.
        
        Args:
            model_path: Path to pretrained DCP model weights (.t7 or .pth)
            device: Device to run inference on
            args: Model arguments object (if provided, overrides other params)
            emb_nn: Embedding network type ('pointnet' or 'dgcnn')
            pointer: Pointer type ('identity' or 'transformer')
            head: Head type ('mlp' or 'svd')
            emb_dims: Embedding dimensions
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Use provided args or create from parameters
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
        
        # Create model
        self.model = DCP(args).to(self.device)
        self.model.eval()
        
        # Load weights - handle relative paths
        if not os.path.isabs(model_path):
            # Try relative to current directory first
            if os.path.isfile(model_path):
                model_path = os.path.abspath(model_path)
            # Try relative to script directory
            elif os.path.isfile(osp.join(osp.dirname(__file__), model_path)):
                model_path = osp.join(osp.dirname(__file__), model_path)
            # Try in pretrained directory
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
                f"Please check the path. Available models in pretrained/: {os.listdir(osp.join(osp.dirname(__file__), 'pretrained')) if osp.exists(osp.join(osp.dirname(__file__), 'pretrained')) else 'N/A'}"
            )
    
    @torch.no_grad()
    def get_correspondences(
        self,
        src_points: np.ndarray,
        tgt_points: np.ndarray,
        return_scores: bool = False
    ) -> tuple:
        """
        Get correspondences between source and target point clouds.
        
        Args:
            src_points: Source point cloud (N, 3)
            tgt_points: Target point cloud (M, 3)
            return_scores: If True, also return correspondence scores matrix
        
        Returns:
            src_corr: Source correspondence points (K, 3)
            tgt_corr: Target correspondence points (K, 3)
            scores: (Optional) Correspondence scores matrix (N, M)
        """
        # Convert to torch tensors and add batch dimension
        src_tensor = torch.from_numpy(src_points.astype(np.float32)).T.unsqueeze(0).to(self.device)  # (1, 3, N)
        tgt_tensor = torch.from_numpy(tgt_points.astype(np.float32)).T.unsqueeze(0).to(self.device)  # (1, 3, M)
        
        # Get embeddings
        src_embedding = self.model.emb_nn(src_tensor)  # (1, emb_dims, N)
        tgt_embedding = self.model.emb_nn(tgt_tensor)  # (1, emb_dims, M)
        
        # Apply pointer/attention
        src_embedding_p, tgt_embedding_p = self.model.pointer(src_embedding, tgt_embedding)
        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p
        
        # Compute correspondence scores (same as SVDHead)
        d_k = src_embedding.size(1)
        scores = torch.matmul(
            src_embedding.transpose(2, 1).contiguous(),  # (1, N, emb_dims)
            tgt_embedding  # (1, emb_dims, M)
        ) / np.sqrt(d_k)  # (1, N, M)
        scores = torch.softmax(scores, dim=2)  # Correspondence matrix (1, N, M)
        
        # Remove batch dimension for easier processing
        scores = scores.squeeze(0)  # (N, M)
        
        # Mutual top-K correspondence extraction (like GeoTransformer)
        topk = 60  # Number of top correspondences to consider from each side
        
        # 1. Source -> Target: For each src point, get top-K target points
        src_topk_scores, src_topk_indices = scores.topk(k=topk, dim=1)  # (N, K)
        src_corr_mat = torch.zeros_like(scores, dtype=torch.bool)
        src_indices = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1).expand(-1, topk)  # (N, K)
        src_corr_mat[src_indices, src_topk_indices] = True  # (N, M)
        
        # 2. Target -> Source: For each tgt point, get top-K source points
        tgt_topk_scores, tgt_topk_indices = scores.topk(k=topk, dim=0)  # (K, M)
        tgt_corr_mat = torch.zeros_like(scores, dtype=torch.bool)
        tgt_indices = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).expand(topk, -1)  # (K, M)
        tgt_corr_mat[tgt_topk_indices, tgt_indices] = True  # (N, M)
        
        # 3. Mutual: Keep only correspondences that are in top-K from both sides
        mutual_corr_mat = torch.logical_and(src_corr_mat, tgt_corr_mat)  # (N, M)
        
        # 4. Extract correspondence indices
        src_corr_indices, tgt_corr_indices = torch.nonzero(mutual_corr_mat, as_tuple=True)  # (K,), (K,)
        
        # 5. Get correspondence scores
        corr_scores = scores[src_corr_indices, tgt_corr_indices]  # (K,)
        
        # 6. Filter by minimum confidence if needed
        min_confidence = 0.005
        valid_mask = corr_scores > min_confidence
        src_corr_indices = src_corr_indices[valid_mask]
        tgt_corr_indices = tgt_corr_indices[valid_mask]
        corr_scores = corr_scores[valid_mask]
        
        # 7. Convert to numpy and get actual points
        src_corr_indices_np = src_corr_indices.cpu().numpy()
        tgt_corr_indices_np = tgt_corr_indices.cpu().numpy()
        
        src_points_np = src_tensor.squeeze(0).T.cpu().numpy()  # (N, 3)
        tgt_points_np = tgt_tensor.squeeze(0).T.cpu().numpy()  # (M, 3)
        
        src_corr_np = src_points_np[src_corr_indices_np]  # (K, 3)
        tgt_corr_np = tgt_points_np[tgt_corr_indices_np]  # (K, 3)
        scores_np = scores.cpu().numpy()  # (N, M)
        corr_scores_np = corr_scores.cpu().numpy()  # (K,)
        
        if return_scores:
            return src_corr_np, tgt_corr_np, scores_np
        else:
            return src_corr_np, tgt_corr_np
    
    @torch.no_grad()
    def get_transformation(
        self,
        src_points: np.ndarray,
        tgt_points: np.ndarray
    ) -> tuple:
        """
        Get transformation matrix (rotation and translation) between point clouds.
        
        Args:
            src_points: Source point cloud (N, 3)
            tgt_points: Target point cloud (M, 3)
        
        Returns:
            rotation: Rotation matrix (3, 3)
            translation: Translation vector (3,)
        """
        src_tensor = torch.from_numpy(src_points.astype(np.float32)).T.unsqueeze(0).to(self.device)
        tgt_tensor = torch.from_numpy(tgt_points.astype(np.float32)).T.unsqueeze(0).to(self.device)
        
        rotation, translation, _, _ = self.model(src_tensor, tgt_tensor)
        
        # Remove batch dimension and convert to numpy
        rotation_np = rotation.squeeze(0).cpu().numpy()  # (3, 3)
        translation_np = translation.squeeze(0).cpu().numpy()  # (3,)
        
        return rotation_np, translation_np


def get_correspondences(
    inference_model: DCPInference,
    point_cloud: np.ndarray,
    rotation_matrix: np.ndarray,
    return_scores: bool = False
) -> tuple:
    """
    Get correspondences between a point cloud and its rotation.
    
    Args:
        inference_model: DCPInference instance
        point_cloud: Input point cloud (N, 3)
        rotation_matrix: Rotation matrix (3, 3)
        return_scores: If True, also return correspondence scores
    
    Returns:
        base: Base point cloud (N, 3)
        rotated: Rotated point cloud (N, 3)
        base_corr: Correspondence points from base (K, 3)
        rotated_corr: Correspondence points from rotated (K, 3)
        scores: (Optional) Correspondence scores matrix
    """
    base = point_cloud.astype(np.float32)
    rotated = (rotation_matrix.astype(np.float32) @ base.T).T
    
    if return_scores:
        base_corr, rotated_corr, scores = inference_model.get_correspondences(
            base, rotated, return_scores=True
        )
        return base, rotated, base_corr, rotated_corr, scores
    else:
        base_corr, rotated_corr = inference_model.get_correspondences(
            base, rotated, return_scores=False
        )
        return base, rotated, base_corr, rotated_corr


def generate_toy_dataset(num_samples: int = 10, num_points: int = 1000, seed: int = 42) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """
    Generate toy dataset with 5-fold rotational symmetry (pentagonal cone).
    
    Args:
        num_samples: Number of samples to generate (default: 10)
        num_points: Number of points per point cloud (default: 1000)
        seed: Random seed
        
    Returns:
        point_clouds: Array of point clouds (num_samples, num_points, 3)
        labels: List of GT rotation parameters for each point cloud
                Each label is [axis_x, axis_y, axis_z, angle] (axis is unit vector)
    """
    np.random.seed(seed)
    
    # Generate base pentagonal cone (5-fold symmetry around z-axis)
    def generate_pentagonal_cone(n_points: int) -> np.ndarray:
        # Create points on a pentagonal cone
        points = []
        for i in range(n_points):
            # Random height
            z = np.random.uniform(-1.0, 1.0)
            # Random angle (will be quantized to pentagon)
            angle_base = np.random.uniform(0, 2 * np.pi)
            # Quantize to 5-fold symmetry
            angle = np.round(angle_base / (2 * np.pi / 5)) * (2 * np.pi / 5)
            # Radius decreases with height
            radius = (1.0 - abs(z)) * 0.5
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y, z])
        points = np.array(points, dtype=np.float32)
        # Center at origin (중점을 원점으로)
        centroid = points.mean(axis=0)
        points = points - centroid
        return points
    
    point_clouds = []
    labels = []
    
    for i in range(num_samples):
        # Generate base cone
        base_cone = generate_pentagonal_cone(num_points)
        
        # Apply random rotation and translation
        # Generate random rotation matrix using QR decomposition
        random_matrix = np.random.randn(3, 3).astype(np.float32)
        Q, R_qr = np.linalg.qr(random_matrix)
        if np.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        random_rot = Q.astype(np.float32)
        random_trans = np.random.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
        
        # Transform point cloud
        transformed_cone = (random_rot @ base_cone.T).T + random_trans
        point_clouds.append(transformed_cone)
        
        # Generate GT rotation axes (5 rotations around transformed z-axis)
        # Original z-axis in transformed space
        z_axis_original = np.array([0, 0, 1], dtype=np.float32)
        z_axis_transformed = (random_rot @ z_axis_original).astype(np.float32)
        z_axis_transformed = z_axis_transformed / (np.linalg.norm(z_axis_transformed) + 1e-8)
        
        # 5-fold rotation: angles are 72, 144, 216, 288, 360 degrees
        gt_rotations = []
        for k in range(5):
            angle_deg = 72.0 * (k + 1)  # 72, 144, 216, 288, 360
            angle_rad = np.deg2rad(angle_deg)
            # Store as [axis_x, axis_y, axis_z, angle_rad]
            gt_rotations.append(np.concatenate([z_axis_transformed, [angle_rad]]))
        
        labels.append(gt_rotations)
    
    return np.array(point_clouds), labels


@torch.no_grad()
def _rotations_from_pairs(base_corr: np.ndarray, rotated_corr: np.ndarray,
                          point_cloud: torch.Tensor, device: torch.device,
                          inlier_dist_threshold: float = 0.01, num_iterations: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RANSAC-style rotation extraction from correspondence pairs.
    Uses 2 pairs of correspondences to compute rotation axis and angle.
    
    Args:
        base_corr: Base correspondences (N, 3)
        rotated_corr: Rotated correspondences (N, 3)
        point_cloud: Original point cloud centered at origin, torch tensor (M, 3)
        device: Torch device
        inlier_dist_threshold: Distance threshold for correspondence pair to be inlier
        num_iterations: Number of rotations to extract
    
    Returns arrays of rotation axes, angles, and chamfer distances.
    """
    if base_corr.shape[0] < 2 or rotated_corr.shape[0] < 2:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    
    n_pairs = base_corr.shape[0]
    
    # Compute rotation from pairs of correspondences
    result_axes, result_angles, result_chamfers = [], [], []
    
    # Try random pairs of correspondences
    max_pairs = min(num_iterations, n_pairs * (n_pairs - 1) // 2)
    pair_indices = []
    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            pair_indices.append((i, j))
    
    # Randomly sample pairs
    if len(pair_indices) > max_pairs:
        import random
        random.seed(42)
        pair_indices = random.sample(pair_indices, max_pairs)
    
    for i, j in pair_indices:
        
        # Get two correspondence pairs
        p1 = base_corr[i]
        p1_rot = rotated_corr[i]
        p2 = base_corr[j]
        p2_rot = rotated_corr[j]
        
        # Compute rotation axis and angle from two pairs
        # Method: Use cross product of vectors to find rotation axis
        v1 = p1_rot - p1
        v2 = p2_rot - p2
        
        # If vectors are parallel, skip
        if np.linalg.norm(np.cross(v1, v2)) < 1e-6:
            continue
        
        # Compute rotation using SVD (more stable)
        # We want R such that R @ p1 = p1_rot and R @ p2 = p2_rot
        # Use Kabsch algorithm with two points
        P = np.stack([p1, p2])  # (2, 3)
        Q = np.stack([p1_rot, p2_rot])  # (2, 3)
        
        # Center points
        P_centered = P - P.mean(axis=0)
        Q_centered = Q - Q.mean(axis=0)
        
        # Compute rotation matrix
        H = P_centered.T @ Q_centered
        U, S, Vt = np.linalg.svd(H)
        R_est = Vt.T @ U.T
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_est) < 0:
            Vt[-1, :] *= -1
            R_est = Vt.T @ U.T
        
        # Extract axis and angle from rotation matrix
        # R = I + sin(θ)[k]× + (1-cos(θ))[k]×²
        # where [k]× is skew-symmetric matrix of axis k
        trace = np.trace(R_est)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if angle < 1e-6:
            continue
        
        # Compute axis from skew-symmetric part
        K = (R_est - R_est.T) / (2 * np.sin(angle))
        axis = np.array([K[2, 1], K[0, 2], K[1, 0]], dtype=np.float32)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        
        # Compute translation (center of rotation)
        # For rotation symmetry, translation is the point on axis closest to centroid
        # Simplified: use origin-centered assumption
        t_est = np.zeros(3, dtype=np.float32)
        
        # Verify rotation with chamfer distance
        point_cloud_np = point_cloud.cpu().numpy()
        axis_torch = torch.from_numpy(axis).to(device)
        angle_torch = torch.tensor(angle, device=device)
        t_torch = torch.from_numpy(t_est).to(device)
        
        # Build rotation matrix from axis-angle
        cos_a = torch.cos(angle_torch)
        sin_a = torch.sin(angle_torch)
        K = torch.zeros(3, 3, device=device)
        K[0, 1] = -axis_torch[2]
        K[0, 2] = axis_torch[1]
        K[1, 0] = axis_torch[2]
        K[1, 2] = -axis_torch[0]
        K[2, 0] = -axis_torch[1]
        K[2, 1] = axis_torch[0]
        
        R_torch = torch.eye(3, device=device) + sin_a * K + (1 - cos_a) * torch.outer(axis_torch, axis_torch)
        
        # Apply rotation (point cloud is already centered)
        pc_rotated = (R_torch @ point_cloud.T).T
        
        chamfer_val, _ = chamfer_distance(
            point_cloud.unsqueeze(0), pc_rotated.unsqueeze(0)
        )
        
        if chamfer_val.item() < 0.1:  # Reasonable threshold for rotation symmetry
            result_axes.append(axis)
            result_angles.append(angle)
            result_chamfers.append(chamfer_val.item())
    
    if len(result_axes) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    
    return (
        np.array(result_axes, dtype=np.float32),
        np.array(result_angles, dtype=np.float32),
        np.array(result_chamfers, dtype=np.float32),
    )


@torch.no_grad()
def _planes_from_pairs(ref_corr: np.ndarray, src_corr: np.ndarray, reflection_matrix: np.ndarray,
                       point_cloud: torch.Tensor, device: torch.device,
                       inlier_dist_threshold: float = 0.01, num_iterations: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    RANSAC-style plane extraction from correspondence pairs.
    
    Args:
        ref_corr: Reference correspondences (N, 3)
        src_corr: Source correspondences (N, 3)
        reflection_matrix: Reflection matrix (3, 3)
        point_cloud: Original point cloud centered at origin, torch tensor (M, 3)
        device: Torch device
        inlier_dist_threshold: Distance threshold for correspondence pair to be inlier
        num_iterations: Number of planes to extract (default: 10)
    
    Returns arrays of plane normals, offsets, centers, and chamfer distances.
    """
    if ref_corr.shape[0] < 3 or src_corr.shape[0] < 3:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    
    # Map src correspondences back to original space
    M_src = (reflection_matrix.astype(np.float32) @ src_corr.T).T  # (N,3)
    
    # Compute per-pair normal and midpoint for ALL pairs
    diff = ref_corr - M_src
    norms = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
    n_per_pair = diff / norms               # unit normal for each pair (N, 3)
    c_per_pair = 0.5 * (ref_corr + M_src)   # midpoint for each pair (N, 3)
    d_per_pair = np.sum(n_per_pair * c_per_pair, axis=1)  # plane offset for each pair (N,)
    
    # Compute inlier matrix in batches: inlier_matrix[i, j] = is pair j inlier to plane i?
    n_pairs = n_per_pair.shape[0]
    n_batch = torch.from_numpy(n_per_pair).to(device, dtype=torch.float32)  # (N, 3)
    d_batch = torch.from_numpy(d_per_pair).to(device, dtype=torch.float32)  # (N,)
    
    ref_corr_torch = torch.from_numpy(ref_corr).to(device, dtype=torch.float32)  # (N, 3)
    M_src_torch = torch.from_numpy(M_src).to(device, dtype=torch.float32)  # (N, 3)
    
    # Compute inlier matrix in batches to avoid OOM
    inlier_matrix = torch.zeros((n_pairs, n_pairs), dtype=torch.bool, device=device)
    batch_size = 128  # Process 128 planes at a time (reduced to save memory)
    
    for start_idx in range(0, n_pairs, batch_size):
        end_idx = min(start_idx + batch_size, n_pairs)
        
        n_chunk = n_batch[start_idx:end_idx]  # (B, 3)
        d_chunk = d_batch[start_idx:end_idx]  # (B,)
        
        # Reflection matrices for this batch
        H_chunk = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0) - 2.0 * torch.bmm(
            n_chunk.unsqueeze(2), n_chunk.unsqueeze(1)
        )  # (B, 3, 3)
        
        shifts = d_chunk.unsqueeze(1) * n_chunk  # (B, 3)
        ref_shifted = ref_corr_torch.unsqueeze(0) - shifts.unsqueeze(1)  # (B, N, 3)
        M_src_shifted = M_src_torch.unsqueeze(0) - shifts.unsqueeze(1)  # (B, N, 3)
        
        ref_reflected = torch.bmm(ref_shifted, H_chunk)  # (B, N, 3)
        M_src_reflected = torch.bmm(M_src_shifted, H_chunk)  # (B, N, 3)
        
        # Distance from reflected ref to M_src and reflected M_src to ref
        dist1 = torch.norm(ref_reflected - M_src_shifted, dim=2)  # (B, N)
        dist2 = torch.norm(M_src_reflected - ref_shifted, dim=2)  # (B, N)
        avg_dist = (dist1 + dist2) / 2.0  # (B, N)
        
        inlier_matrix[start_idx:end_idx, :] = avg_dist < inlier_dist_threshold  # (B, N)
    
    # RANSAC iterations
    available_mask = np.ones(n_pairs, dtype=bool)
    result_normals, result_ds, result_centers = [], [], []
    
    for _ in range(num_iterations):
        if not available_mask.any() or available_mask.sum() < 10:
            break
        
        # Find plane with maximum inliers among available pairs
        available_inliers = inlier_matrix[:, available_mask].sum(dim=1).cpu().numpy()
        best_idx = np.argmax(available_inliers)
        
        if available_inliers[best_idx] < 10:
            break
        
        # Get inlier pairs (only from available pairs)
        inlier_local = inlier_matrix[best_idx, :].cpu().numpy() & available_mask
        
        # Extract inlier points
        X_inliers = M_src[inlier_local]
        Y_inliers = ref_corr[inlier_local]
        
        # Compute optimal plane from inliers (same as previous code)
        diffs = Y_inliers - X_inliers
        diffs_centered = diffs - diffs.mean(axis=0)
        cov = diffs_centered.T @ diffs_centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_dir = eigvecs[:, np.argmax(eigvals)]
        n_vec = principal_dir / (np.linalg.norm(principal_dir) + 1e-8)
        
        # Plane passes through average midpoint
        midpoints = c_per_pair[inlier_local]
        mean_mid = midpoints.mean(axis=0)
        d_val = -float(n_vec.dot(mean_mid))
        
        result_normals.append(n_vec.astype(np.float32))
        result_ds.append(np.float32(d_val))
        result_centers.append(mean_mid.astype(np.float32))
        
        # Mark inliers as used
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
    
    # Compute chamfer distances
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


@torch.no_grad()
def evaluate_rotation_symmetry_by_correspondence(
    point_clouds: np.ndarray,
    gt_labels: List[List[np.ndarray]],
    inference_model: DCPInference,
    dist_threshold: float = 0.1,
    chamfer_threshold: float = 0.01,
) -> Tuple[float, float, float, float]:
    """
    Evaluate rotation symmetry detection using DCP correspondences.
    
    Args:
        point_clouds: Array of point clouds (N, M, 3)
        gt_labels: List of GT rotation parameters [axis_x, axis_y, axis_z, angle] for each point cloud
        inference_model: DCPInference instance
        dist_threshold: Distance threshold for matching (axis distance + angle difference)
        chamfer_threshold: Chamfer distance threshold for filtering
        
    Returns:
        precision, recall, f1, avg_chamfer
    """
    device = inference_model.device
    precision_sum = 0.0
    recall_sum = 0.0
    low_chamfer_sum = 0.0
    low_chamfer_count = 0
    total_samples = len(point_clouds)
    total_pred = 0

    # logging: save idx, axis, angle, chamfer, min distance to GT
    gt_lines: List[str] = []
    out_gt_path = osp.join(osp.dirname(__file__), 'dcp_rotation_symmetry.txt')

    pbar = tqdm(range(total_samples), desc='Evaluating samples', dynamic_ncols=True)
    for i in pbar:
        pc = point_clouds[i].astype(np.float32)
        
        # Extract GT labels: [axis_x, axis_y, axis_z, angle]
        gt_params_list = gt_labels[i]
        gt_axes_list = [p[:3] for p in gt_params_list]
        gt_angles_list = [float(p[3]) for p in gt_params_list]
        
        gt_axes_tensor = torch.stack([torch.tensor(a, dtype=torch.float32, device=device) for a in gt_axes_list])
        gt_axes_tensor = gt_axes_tensor / (torch.norm(gt_axes_tensor, dim=1, keepdim=True) + 1e-8)
        gt_angles_tensor = torch.tensor(gt_angles_list, dtype=torch.float32, device=device)  # (num_gt,)

        pred_axes_torch: List[torch.Tensor] = []
        pred_angles: List[float] = []
        pred_chamfers: List[float] = []
        rotations_kept = 0

        # Use original point cloud directly
        base = pc.astype(np.float32)

        # Try multiple rotation angles to find symmetry (72, 144, 216, 288 degrees)
        # For 5-fold symmetry, we test rotations around estimated axes
        test_angles = [np.deg2rad(72), np.deg2rad(144), np.deg2rad(216), np.deg2rad(288)]
        
        all_base_corr = []
        all_rotated_corr = []
        
        for test_angle in test_angles:
            # Create rotation matrix around z-axis (will be transformed)
            cos_a = np.cos(test_angle)
            sin_a = np.sin(test_angle)
            R_test = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Get correspondences
            _, rotated, base_corr, rotated_corr = get_correspondences(
                inference_model, base, R_test
            )
            
            if base_corr.shape[0] > 0:
                all_base_corr.append(base_corr)
                all_rotated_corr.append(rotated_corr)
        
        # Combine all correspondences
        if len(all_base_corr) > 0:
            base_corr_all = np.concatenate(all_base_corr, axis=0)
            rotated_corr_all = np.concatenate(all_rotated_corr, axis=0)
        else:
            base_corr_all = np.zeros((0, 3), dtype=np.float32)
            rotated_corr_all = np.zeros((0, 3), dtype=np.float32)
        
        num_correspondences = base_corr_all.shape[0]
        
        # Centerize point cloud at origin
        X = torch.from_numpy(base).float().to(device)
        X_center = X.mean(dim=0)
        X_orig = X - X_center  # Center at origin

        # Get rotations from correspondence pairs
        axes_np, angles_np, chamfer_np = _rotations_from_pairs(
            base_corr_all, rotated_corr_all, X_orig, device,
            inlier_dist_threshold=0.01, num_iterations=100
        )

        # Process rotations if any found
        if axes_np.shape[0] > 0:
            # Convert to torch tensors
            axes_batch = torch.from_numpy(axes_np).to(device)  # (K, 3)
            angles_batch = torch.from_numpy(angles_np).to(device, dtype=torch.float32)  # (K,)
            chamfer_batch = torch.from_numpy(chamfer_np).to(device, dtype=torch.float32)  # (K,)
            
            # Filter rotations by chamfer distance threshold
            filter_mask = chamfer_batch < chamfer_threshold
            
            if filter_mask.any():
                pred_axes_torch = axes_batch[filter_mask].unbind(0)
                pred_angles = angles_batch[filter_mask].tolist()
                pred_chamfers = chamfer_batch[filter_mask].tolist()
            else:
                pred_axes_torch = []
                pred_angles = []
                pred_chamfers = []

        if len(pred_axes_torch) == 0:
            precision_sum += 1.0  # No predictions = perfect precision (no false positives)
        else:
            # Keep data on GPU for efficiency
            keep_axes = torch.stack(pred_axes_torch)  # (N, 3)
            keep_angles = torch.tensor(pred_angles, dtype=torch.float32, device=device)  # (N,)
            keep_chamfers = torch.tensor(pred_chamfers, dtype=torch.float32, device=device)  # (N,)
            rotations_kept = keep_axes.shape[0]

            # Track low-chamfer stats
            low_chamfer_sum += float(keep_chamfers.sum().item())
            low_chamfer_count += len(keep_chamfers)

            # Compute GT distance metrics
            # Axis distance: (num_pred, num_gt)
            pred_axes_expanded = keep_axes.unsqueeze(1)  # (num_pred, 1, 3)
            gt_axes_expanded = gt_axes_tensor.unsqueeze(0)  # (1, num_gt, 3)
            
            # Axis can point in either direction (sign-invariant)
            dist_pos = torch.norm(pred_axes_expanded - gt_axes_expanded, dim=2)
            dist_neg = torch.norm(pred_axes_expanded + gt_axes_expanded, dim=2)
            axis_dist = torch.minimum(dist_pos, dist_neg)  # (num_pred, num_gt)
            
            # Angle distance: wrap around 2π
            pred_angles_expanded = keep_angles.unsqueeze(1)  # (num_pred, 1)
            gt_angles_expanded = gt_angles_tensor.unsqueeze(0)  # (1, num_gt)
            
            angle_diff = torch.abs(pred_angles_expanded - gt_angles_expanded)
            angle_diff = torch.minimum(angle_diff, 2 * np.pi - angle_diff)  # Wrap around
            
            # Combined metric: axis distance + normalized angle difference
            angle_weight = 0.1  # Weight for angle difference
            metric = axis_dist + angle_weight * angle_diff / np.pi
            
            # Match matrix: (num_pred, num_gt)
            match_matrix = metric < dist_threshold
            
            # Precision: TP / #predicted
            matches_per_pred = match_matrix.any(dim=1)  # (num_pred,)
            tp_precision = int(matches_per_pred.sum().item())
            precision_sum += tp_precision / keep_axes.shape[0]
            
            # Recall: TP / #GT
            matches_per_gt = match_matrix.any(dim=0)  # (num_gt,)
            tp_recall = int(matches_per_gt.sum().item())
            recall_sum += tp_recall / len(gt_axes_tensor)
            
            # Logging: compute min GT distance for each pred rotation
            min_gt_dist_per_pred = metric.min(dim=1)[0]  # (num_pred,)
            
            for k in range(keep_axes.shape[0]):
                axis = keep_axes[k]
                angle = keep_angles[k].item()
                chamfer_val = keep_chamfers[k].item()
                min_metric = min_gt_dist_per_pred[k].item()
                gt_lines.append(
                    f"{i}\t{float(axis[0]):.6f}\t{float(axis[1]):.6f}\t{float(axis[2]):.6f}\t{angle:.6f}\t{chamfer_val:.6f}\t{min_metric:.6f}"
                )

            total_pred += len(pred_chamfers)

        # update progress postfix
        done = i + 1
        pbar.set_postfix(
            corr=num_correspondences,
            rotations=rotations_kept,
            P=(precision_sum / done) if done > 0 else 0.0,
            R=(recall_sum / done) if done > 0 else 0.0,
        )

    # write log file
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
    parser = argparse.ArgumentParser(description='DCP Rotation Symmetry Evaluation')
    # Default to pretrained model if available
    default_model = None
    pretrained_dir = osp.join(osp.dirname(__file__), 'pretrained')
    if osp.exists(pretrained_dir):
        if osp.exists(osp.join(pretrained_dir, 'dcp_v2.t7')):
            default_model = osp.join(pretrained_dir, 'dcp_v2.t7')
        elif osp.exists(osp.join(pretrained_dir, 'dcp_v1.t7')):
            default_model = osp.join(pretrained_dir, 'dcp_v1.t7')
    
    parser.add_argument('--model_path', type=str, default=default_model,
                        help=f'Path to DCP model weights (default: {default_model if default_model else "required"})')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of toy dataset samples to generate')
    parser.add_argument('--num_points', type=int, default=1000,
                        help='Number of points per point cloud')
    parser.add_argument('--dist_threshold', type=float, default=0.1,
                        help='Distance threshold for matching')
    parser.add_argument('--chamfer_threshold', type=float, default=0.01,
                        help='Chamfer distance threshold')
    parser.add_argument('--emb_nn', type=str, default='dgcnn',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding network type')
    parser.add_argument('--pointer', type=str, default='transformer',
                        choices=['identity', 'transformer'],
                        help='Pointer type')
    parser.add_argument('--head', type=str, default='svd',
                        choices=['mlp', 'svd'],
                        help='Head type')
    parser.add_argument('--emb_dims', type=int, default=512,
                        help='Embedding dimensions')
    
    args = parser.parse_args()
    
    # Check if model path is provided
    if args.model_path is None:
        parser.error("--model_path is required. Available models: pretrained/dcp_v1.t7, pretrained/dcp_v2.t7")
    
    # Create inference model
    infer = DCPInference(
        args.model_path, 
        device=args.device,
        emb_nn=args.emb_nn,
        pointer=args.pointer,
        head=args.head,
        emb_dims=args.emb_dims
    )
    
    # Generate toy dataset
    print(f"Generating toy dataset with {args.num_samples} samples, {args.num_points} points each...")
    pcs, labels = generate_toy_dataset(num_samples=args.num_samples, num_points=args.num_points)
    print(f"Generated {len(pcs)} point clouds with {len(labels)} GT labels")

    # Save first point cloud for viewing (CloudCompare, MeshLab 등으로 확인 가능)
    sample_path = osp.join(osp.dirname(__file__), 'pentagonal_cone_sample.ply')
    with open(sample_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pcs[0].shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pcs[0]:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"Saved sample for viewing: {sample_path}")
    
    # Evaluate
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

