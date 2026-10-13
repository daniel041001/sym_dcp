"""Step (1) Correspondence extraction.

PareNetMatcher wraps the pretrained PARE-Net (3DMatch) registration model as the matcher M(X, X').
  reflection: M(X, M_ref X)  -> map the mirrored side back by the involution -> intra-shape pairs
  rotation:   M(X, X) with coarse self-masking, then drop pairs closer than min_radius
"""
from __future__ import annotations

import importlib.util
import os.path as osp
import sys
from typing import Any, Optional, Tuple

import numpy as np
import torch

from .config import MatcherConfig

DCP_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


def _recursive_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _recursive_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_to_device(v, device) for v in obj]
    return obj


def _load_parenet(parenet_root: str):
    exp3d = osp.join(parenet_root, "experiments", "3DMatch")
    for p in (parenet_root, exp3d):
        if p not in sys.path:
            sys.path.insert(0, p)
    mods = []
    for name, fn in (("parenet_refl_3dmatch_config", "config.py"), ("parenet_refl_3dmatch_model", "model.py")):
        spec = importlib.util.spec_from_file_location(name, osp.join(exp3d, fn))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mods.append(mod)
    return mods[0].make_cfg, mods[1].create_model


class PareNetMatcher:
    """Pretrained registration-based matcher M(.,.) returning putative correspondences."""

    def __init__(self, cfg: MatcherConfig = MatcherConfig()):
        self.device = torch.device(cfg.device)
        if self.device.type != "cuda":
            raise RuntimeError("PARE-Net requires CUDA (pareconv hardcodes .cuda()).")
        torch.cuda.set_device(self.device)
        make_cfg, create_model = _load_parenet(osp.join(DCP_ROOT, "parenet"))
        from pareconv.utils.data import precompute_neibors, registration_collate_fn_stack_mode

        self._collate, self._neighbors = registration_collate_fn_stack_mode, precompute_neibors
        self._cfg = make_cfg()
        self._cfg.backbone.init_voxel_size = float(cfg.init_voxel_size)
        self._cfg.fine_matching.num_hypotheses = int(cfg.num_correspondences)
        self._model = create_model(self._cfg).to(self.device)
        weights = cfg.weights if osp.isabs(cfg.weights) else osp.join(DCP_ROOT, cfg.weights)
        ckpt = torch.load(weights, map_location=self.device)
        self._model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=False)
        self._model.eval()
        self._self_mask = None

    @torch.no_grad()
    def match(self, src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ref = np.asarray(src, dtype=np.float32)
        tgt = np.asarray(tgt, dtype=np.float32)
        data = {
            "ref_points": ref, "src_points": tgt,
            "ref_feats": np.ones_like(ref[:, :1]), "src_feats": np.ones_like(tgt[:, :1]),
            "transform": np.eye(4, dtype=np.float32),
        }
        bb = self._cfg.backbone
        col = _recursive_to_device(self._collate([data], bb.num_stages, bb.init_voxel_size, bb.num_neighbors, bb.subsample_ratio), self.device)
        col.update(_recursive_to_device(self._neighbors(col["points"], col["lengths"], bb.num_stages, bb.num_neighbors), self.device))
        out = self._model(col)
        a, b = out["ref_corr_points"], out["src_corr_points"]
        a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
        b = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else b
        return np.asarray(a, np.float32), np.asarray(b, np.float32)

    # ---- coarse-level self-match masking (rotation) ----
    def install_self_mask(self, radius: float) -> None:
        if self._self_mask is not None:
            return
        wrapper = _SelfMaskedSuperPointMatching(self._model.coarse_matching, radius)
        self._model.coarse_matching = wrapper

        def _pre_hook(module, args):
            d = args[0]
            n_ref = int(d["lengths"][-1][0].item())
            pts = d["points"][-1]
            wrapper.ref_pts, wrapper.src_pts = pts[:n_ref], pts[n_ref:]

        self._model.register_forward_pre_hook(_pre_hook)
        self._self_mask = wrapper


class _SelfMaskedSuperPointMatching(torch.nn.Module):
    """Zero coarse scores of spatially adjacent superpoint pairs before dual normalization,
    so the correspondence budget goes to non-local (symmetric) pairs."""

    def __init__(self, inner: torch.nn.Module, radius: float):
        super().__init__()
        self.inner, self.radius = inner, float(radius)
        self.ref_pts: Optional[torch.Tensor] = None
        self.src_pts: Optional[torch.Tensor] = None

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        from pareconv.modules.ops import pairwise_distance
        if ref_masks is None:
            ref_masks = torch.ones((ref_feats.shape[0],), dtype=torch.bool, device=ref_feats.device)
        if src_masks is None:
            src_masks = torch.ones((src_feats.shape[0],), dtype=torch.bool, device=src_feats.device)
        ri = torch.nonzero(ref_masks, as_tuple=True)[0]
        si = torch.nonzero(src_masks, as_tuple=True)[0]
        scores = torch.exp(-pairwise_distance(ref_feats[ri], src_feats[si], normalized=True))
        if self.ref_pts is not None and self.src_pts is not None:
            scores = scores.masked_fill(torch.cdist(self.ref_pts[ri], self.src_pts[si]) < self.radius, 0.0)
        if self.inner.dual_normalization:
            r = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
            c = scores / scores.sum(dim=0, keepdim=True).clamp_min(1e-12)
            scores = r * c
        k = int(min(self.inner.num_correspondences, int((scores > 0).sum().item())))
        if k <= 0:
            k = int(min(self.inner.num_correspondences, scores.numel()))
        cs, idx = scores.view(-1).topk(k=k, largest=True)
        return ri[idx // scores.shape[1]], si[idx % scores.shape[1]], cs


def mirror_matrix(axis: int = 2) -> np.ndarray:
    M = np.eye(3, dtype=np.float32)
    M[axis, axis] = -1.0
    return M


def reflection_correspondences(matcher: PareNetMatcher, X: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Match X with its mirrored copy M X; return intra-shape pairs (x_i, x_j) in X's frame."""
    base = X.astype(np.float32)
    x, x_bar = matcher.match(base, (M.astype(np.float32) @ base.T).T)
    if x.shape[0] == 0 or x_bar.shape[0] == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return x, (M.astype(np.float32) @ x_bar.T).T          # x_bar_j = M x_j  ->  x_j = M x_bar_j


def rotation_correspondences(matcher: PareNetMatcher, X: np.ndarray, min_radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Self-match X (self-mask must be installed); drop near-identity pairs."""
    base = X.astype(np.float32)
    x, y = matcher.match(base, base)
    if x.shape[0] > 0 and y.shape[0] > 0:
        keep = np.linalg.norm(x - y, axis=1) >= min_radius
        x, y = x[keep], y[keep]
    if y.shape[0] == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return x, y
