"""Datasets and metrics. The metrics are copied unchanged from reflection_final / rotation_final:

reflection: plane distance sqrt(min(|n-n*|,|n+n*|)^2 + |d -/+ d*|^2) < 0.05 k, k = 1..4, P/R averaged per sample.
rotation:   axis distance min(|a-a*|,|a+a*|) < tau, tau in {.05,.10,.15,.20}, and line distance to the GT axis < 0.1.
"""
from __future__ import annotations

import os.path as osp
import pickle
from typing import List, Tuple

import numpy as np
import torch

DCP_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


# ======================================================================
# Data
# ======================================================================
def load_reflection_dataset(root: str = "data/preprocessed_data_r"):
    root = root if osp.isabs(root) else osp.join(DCP_ROOT, root)
    with open(osp.join(root, "point_clouds.pkl"), "rb") as f:
        pcs = pickle.load(f)
    with open(osp.join(root, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    return pcs, labels


def load_rotation_dataset(path: str = "data/shapenet_rot_gt_cache.pkl"):
    path = path if osp.isabs(path) else osp.join(DCP_ROOT, path)
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["point_clouds"], d["gt_labels"]


# ======================================================================
# Reflection metric
# ======================================================================
class ReflectionScorer:
    def __init__(self, device, base: float = 0.05):
        self.device, self.base = device, base
        self.P = self.R = 0.0
        self.n = 0
        self.lines: List[str] = []

    def add(self, i: int, gt_params, n_np: np.ndarray, d_np: np.ndarray, sde_np: np.ndarray, X_center: torch.Tensor):
        """n_np, d_np: planes n.x + d = 0 in the *uncentred* correspondence frame (as returned by detect_reflection)."""
        dev = self.device
        gn = torch.stack([torch.tensor(p[:3], dtype=torch.float32, device=dev) for p in gt_params])
        gn = gn / (torch.norm(gn, dim=1, keepdim=True) + 1e-8)
        gd = torch.tensor([float(p[3]) for p in gt_params], dtype=torch.float32, device=dev) - (gn @ X_center)
        self.n += 1
        if n_np.shape[0] == 0:
            self.P += 1
            return
        pn = torch.from_numpy(n_np).to(dev)
        pd = torch.from_numpy(d_np).to(dev, dtype=torch.float32) - (pn @ X_center)
        dpos = torch.norm(pn.unsqueeze(1) - gn.unsqueeze(0), dim=2)
        dneg = torch.norm(pn.unsqueeze(1) + gn.unsqueeze(0), dim=2)
        mn = torch.minimum(dpos, dneg)
        od = torch.where(dpos < dneg, torch.abs(pd.unsqueeze(1) - gd.unsqueeze(0)), torch.abs(pd.unsqueeze(1) + gd.unsqueeze(0)))
        metric = torch.sqrt(mn * mn + od * od)
        p = r = 0.0
        for k in range(1, 5):
            m = metric < self.base * k
            p += int(m.any(dim=1).sum().item()) / pn.shape[0]
            r += int(m.any(dim=0).sum().item()) / len(gn)
        self.P += p / 4.0
        self.R += r / 4.0
        md = metric.min(dim=1)[0]
        for k in range(pn.shape[0]):
            self.lines.append(f"{i}\t{float(pn[k,0]):.6f}\t{float(pn[k,1]):.6f}\t{float(pn[k,2]):.6f}\t{float(pd[k]):.6f}\t{float(sde_np[k]):.6f}\t{float(md[k]):.6f}")

    def result(self) -> Tuple[float, float, float]:
        P, R = self.P / max(self.n, 1), self.R / max(self.n, 1)
        return P, R, (2 * P * R / (P + R) if P + R > 0 else 0.0)

    header = "idx\tnx\tny\tnz\td\tchamfer\tmin_gt_dist"


# ======================================================================
# Rotation metric
# ======================================================================
def line_to_line_distance(p_pred: torch.Tensor, n_pred: torch.Tensor, n_gt: torch.Tensor) -> torch.Tensor:
    """(K,G) distance between predicted lines p + s n and GT lines through the origin."""
    K, G = p_pred.shape[0], n_gt.shape[0]
    d1 = n_gt.unsqueeze(0).expand(K, G, 3)
    c = torch.cross(d1, n_pred.unsqueeze(1).expand(K, G, 3), dim=2)
    cn = torch.linalg.norm(c, dim=2).clamp_min(1e-8)
    skew = (p_pred.unsqueeze(1).expand(K, G, 3) * c).sum(dim=2).abs() / cn
    par = torch.linalg.norm(torch.cross(p_pred.unsqueeze(1).expand(K, G, 3), d1, dim=2), dim=2)
    return torch.where(cn < 1e-6, par, skew)


class RotationScorer:
    thresholds = (0.05, 0.10, 0.15, 0.20)
    header = "idx\tax\tay\taz\tangle\tchamfer\tmin_gt_dist"

    def __init__(self, device, identity_theta: float = 1e-2):
        self.device, self.identity_theta = device, identity_theta
        self.P = [0.0] * 4
        self.R = [0.0] * 4
        self.n = 0
        self.lines: List[str] = []

    def add(self, i: int, gt_params, axes: np.ndarray, angles: np.ndarray, sde: np.ndarray, pts: np.ndarray):
        dev = self.device
        self.n += 1
        gt = [np.asarray(g).reshape(-1)[:3] for g in gt_params if np.asarray(g).reshape(-1).shape[0] >= 3]
        if not gt:
            self.P = [v + 1.0 for v in self.P]
            return
        G = torch.stack([torch.tensor(a, dtype=torch.float32, device=dev) for a in gt])
        G = G / (torch.norm(G, dim=1, keepdim=True) + 1e-8)
        keep = None
        if axes.shape[0] > 0:
            ang = torch.from_numpy(angles).to(dev, torch.float32)
            m = ang.abs() > self.identity_theta        # (the old chamfer < inf filter is a no-op)
            if m.any():
                keep = m
        if keep is None:
            self.P = [v + 1.0 for v in self.P]
            return
        A = torch.from_numpy(axes).to(dev)[keep]
        PT = torch.from_numpy(pts).to(dev, torch.float32)[keep]
        ad = torch.minimum(torch.norm(A.unsqueeze(1) - G.unsqueeze(0), dim=2), torch.norm(A.unsqueeze(1) + G.unsqueeze(0), dim=2))
        ld = line_to_line_distance(PT, A, G)
        for ti, thr in enumerate(self.thresholds):
            mm = (ad < thr) & (ld < 0.1)
            self.P[ti] += int(mm.any(dim=1).sum().item()) / A.shape[0]
            self.R[ti] += int(mm.any(dim=0).sum().item()) / len(G)
        md = ad.min(dim=1)[0]
        ang_k, sde_k = angles[keep.cpu().numpy()], sde[keep.cpu().numpy()]
        for k in range(A.shape[0]):
            self.lines.append(f"{i}\t{float(A[k,0]):.6f}\t{float(A[k,1]):.6f}\t{float(A[k,2]):.6f}\t{float(ang_k[k]):.6f}\t{float(sde_k[k]):.6f}\t{float(md[k]):.6f}")

    def result(self):
        P = [v / max(self.n, 1) for v in self.P]
        R = [v / max(self.n, 1) for v in self.R]
        F = [2 * p * r / (p + r) if p + r > 0 else 0.0 for p, r in zip(P, R)]
        return P, R, F, float(sum(F) / len(F))
