import os
import os.path as osp

import numpy as np
import torch

from .DCP_base_model import DCP

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