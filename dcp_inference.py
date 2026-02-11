"""
DCP 모델을 사용하여 correspondence를 추출하는 inference 함수
GeoTransformer의 inference_2.py와 동일한 기능 제공
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


def load_preprocessed_dataset(dataset_root: str) -> Tuple[np.ndarray, list]:
    """
    Load preprocessed dataset from directory.
    
    Args:
        dataset_root: Root directory containing preprocessed data
        
    Returns:
        point_clouds: Array of point clouds (N, M, 3)
        labels: List of ground truth labels for each point cloud
    """
    import pickle
    
    # Try to load from pickle file
    pcd_file = osp.join(dataset_root, 'point_clouds.pkl')
    labels_file = osp.join(dataset_root, 'labels.pkl')
    
    if osp.exists(pcd_file) and osp.exists(labels_file):
        with open(pcd_file, 'rb') as f:
            point_clouds = pickle.load(f)
        with open(labels_file, 'rb') as f:
            labels = pickle.load(f)
    else:
        # Try to load individual files
        import glob
        pcd_files = sorted(glob.glob(osp.join(dataset_root, '*.npy')))
        
        if len(pcd_files) == 0:
            raise FileNotFoundError(f"No data found in {dataset_root}")
        
        point_clouds = []
        labels = []
        
        for pcd_file in pcd_files:
            data = np.load(pcd_file, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                # File contains both point cloud and labels
                item = data.item()
                point_clouds.append(item['points'])
                labels.append(item.get('labels', []))
            else:
                # File contains only point cloud
                point_clouds.append(data)
                
                # Try to load corresponding label file
                label_file = pcd_file.replace('.npy', '_labels.npy')
                if osp.exists(label_file):
                    label_data = np.load(label_file, allow_pickle=True)
                    labels.append(label_data.tolist() if hasattr(label_data, 'tolist') else [])
                else:
                    labels.append([])
        
        point_clouds = np.array(point_clouds)
    
    return point_clouds, labels


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
def evaluate_reflection_symmetry_by_correspondence(
    point_clouds: np.ndarray,
    gt_labels: List[List[np.ndarray]],
    reflection_matrices: List[np.ndarray],
    inference_model: DCPInference,
    dist_threshold: float = 0.05,
    chamfer_threshold: float = 1e-3,
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
                                                        inlier_dist_threshold=0.007, num_iterations=100)

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
            
            # Match matrix: (num_pred, num_gt)
            match_matrix = metric < dist_threshold
            
            # Precision: TP / #predicted
            # TP = number of predicted planes that match at least one GT
            matches_per_pred = match_matrix.any(dim=1)  # (num_pred,)
            tp_precision = int(matches_per_pred.sum().item())
            precision_sum += tp_precision / keep_normals.shape[0]
            
            # Recall: TP / #GT
            # TP = number of GT planes that are matched by at least one prediction
            matches_per_gt = match_matrix.any(dim=0)  # (num_gt,)
            tp_recall = int(matches_per_gt.sum().item())
            recall_sum += tp_recall / len(gt_normals_tensor)
            
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


def main():
    parser = argparse.ArgumentParser(description='DCP Reflection Symmetry Evaluation')
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
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to preprocessed data directory (default: preprocessed_data_r)')
    parser.add_argument('--dist_threshold', type=float, default=0.05,
                        help='Distance threshold for matching')
    parser.add_argument('--chamfer_threshold', type=float, default=0.0005,
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
    
    # Load dataset
    if args.data_root is None:
        data_root = osp.join(osp.dirname(__file__), 'preprocessed_data_r')
    else:
        data_root = args.data_root
    
    pcs, labels = load_preprocessed_dataset(data_root)
    
    # Evaluate
    precision, recall, f1, avg_ch = evaluate_reflection_symmetry_by_correspondence(
        pcs, labels, [], infer, 
        dist_threshold=args.dist_threshold, 
        chamfer_threshold=args.chamfer_threshold
    )
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Chamfer:   {avg_ch:.6f}")


if __name__ == '__main__':
    main()

