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