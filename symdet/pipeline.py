"""Sequential detection loops (Algorithm 1): generate -> verify/refine (x2) -> select -> remove -> repeat."""
from __future__ import annotations

import numpy as np
import torch

from .config import ReflectionConfig, RotationConfig
from .generate_hypothesis import (iter_rotation_hypotheses, reflection_hypotheses, reflection_inlier_matrix,
                                  reflection_residual, rodrigues, rotation_residual, translation)
from .refinement import refine_reflection, refine_rotation
from .selection import remove_reflection_support, rotation_removal_mask, select_rotation
from .verification import (min_support, reflection_ratio_ok, reflection_sde, reflection_sde_batch, rotation_gate,
                           rotation_sde)


def _empty_planes():
    return (np.zeros((0, 3), np.float32), np.zeros((0,), np.float32), np.zeros((0, 3), np.float32), np.zeros((0,), np.float32))


def _empty_axes():
    return (np.zeros((0, 3), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.float32), np.zeros((0, 3), np.float32))


# ======================================================================
# Reflection
# ======================================================================
@torch.no_grad()
def detect_reflection(x: np.ndarray, x_prime: np.ndarray, X: torch.Tensor, cfg: ReflectionConfig = ReflectionConfig()):
    """x, x': intra-shape symmetry correspondences. X: point cloud (the evaluator passes it centred).
    Returns normals (K,3), d_val (K,) for planes n.x + d_val = 0, midpoints (K,3), reported SDE (K,)."""
    if x.shape[0] < 3 or x_prime.shape[0] < 3:
        return _empty_planes()
    dev, f32, f64 = X.device, torch.float32, torch.float64

    # (2) hypotheses: one plane per pair + inlier matrix
    n_np, c_np, d_np = reflection_hypotheses(x, x_prime)
    N = n_np.shape[0]
    n32, d32 = torch.from_numpy(n_np).to(dev, f32), torch.from_numpy(d_np).to(dev, f32)
    x32, xp32 = torch.from_numpy(x).to(dev, f32), torch.from_numpy(x_prime).to(dev, f32)
    inlier_matrix = reflection_inlier_matrix(n32, d32, x32, xp32, cfg.inlier_dist)

    # (3a) pool pruning (tau^0)
    valid = reflection_sde_batch(X, n32, d32) < cfg.pool_sde
    if not bool(valid.any()):
        return _empty_planes()

    min_support_abs = N * cfg.min_support_pct / 100.0
    avail = torch.ones(N, dtype=torch.bool, device=dev)
    cand = torch.ones(N, dtype=torch.bool, device=dev)     # soft-kill: a failed hypothesis loses candidacy only
    x64, xp64 = x32.to(f64), xp32.to(f64)
    c64, n64 = torch.from_numpy(c_np).to(dev, f64), torch.from_numpy(n_np).to(dev, f64)
    res_n, res_d, res_c = [], [], []

    for _ in range(cfg.max_steps):
        pool = avail & valid
        n_pool = int(pool.sum().item())
        if n_pool == 0 or n_pool < min_support_abs or not bool((pool & cand).any()):
            break
        # candidate = most supported hypothesis in the remaining pool
        counts = inlier_matrix @ pool.to(f32)
        counts = torch.where(pool & cand, counts, torch.full_like(counts, -1.0))
        best = int(torch.argmax(counts).item())
        if float(counts[best].item()) < min_support_abs:
            break
        inlier = (inlier_matrix[best] > 0) & pool

        # (3b) 1st refinement
        n_vec, mean_mid = n64[best].clone(), c64[best].clone()
        d_val = -(n_vec @ mean_mid)
        n_vec, mean_mid, d_val, inlier = refine_reflection(n_vec, mean_mid, d_val, inlier, pool, x64, xp64, c64, cfg.inlier_dist, cfg.refine_iters)
        inlier = (reflection_residual(n_vec, d_val, x64, xp64) < cfg.inlier_dist) & pool

        # (3a) verification of the refined plane: SDE and inlier ratio
        nv32 = n_vec.to(f32)
        if reflection_sde(X, nv32, d_val.to(f32) * nv32) >= cfg.sde:
            cand[best] = False; continue
        if not reflection_ratio_ok(int(inlier.sum().item()), n_pool, cfg.inlier_ratio_pct):
            cand[best] = False; continue

        # (3b) 2nd refinement
        n_vec, mean_mid, d_val, inlier = refine_reflection(n_vec, mean_mid, d_val, inlier, pool, x64, xp64, c64, cfg.inlier_dist, cfg.refine_iters)

        # (4) accept + remove support
        res_n.append(n_vec.to(f32).cpu().numpy()); res_d.append(np.float32(float(d_val.item()))); res_c.append(mean_mid.to(f32).cpu().numpy())
        remove_reflection_support(inlier, avail, valid, inlier_matrix)

    if not res_n:
        return _empty_planes()
    rn, rd, rc = np.asarray(res_n, np.float32), np.asarray(res_d, np.float32), np.asarray(res_c, np.float32)
    # reported SDE keeps the original reporting convention (X - d_val n); not used for any decision
    sde = np.zeros(len(rn), np.float32)
    for i in range(len(rn)):
        nv, dv = torch.from_numpy(rn[i]).to(dev), torch.tensor(rd[i], device=dev)
        sde[i] = reflection_sde(X, nv, -(dv * nv))
    return rn, rd, rc, sde


# ======================================================================
# Rotation
# ======================================================================
@torch.no_grad()
def detect_rotation(x: np.ndarray, x_prime: np.ndarray, X: torch.Tensor, cfg: RotationConfig = RotationConfig()):
    """x, x': self-correspondences (same frame as X). Returns axes (K,3), angles (K,), SDE (K,), axis points (K,3)."""
    seed = 2
    N = int(x.shape[0])
    if N < seed:
        return _empty_axes()
    dev = X.device
    P_all = torch.from_numpy(x).to(dev, torch.float32)
    Q_all = torch.from_numpy(x_prime).to(dev, torch.float32)
    thr = torch.tensor(cfg.inlier_dist, device=dev, dtype=torch.float32)

    # (2) random 2-pair hypotheses -> (3a) pool pruning (tau^0) -> inlier rows
    H_n, H_p, H_th, H_inl = [], [], [], []
    for n, p, th, R, t in iter_rotation_hypotheses(P_all, Q_all, cfg.num_hypotheses, cfg.batch, seed):
        ok = rotation_sde(X, R, t) < cfg.pool_sde
        if not ok.any():
            continue
        n, p, th, R, t = n[ok], p[ok], th[ok], R[ok], t[ok]
        H_inl.append(rotation_residual(R, t, P_all, Q_all) < thr)
        H_n.append(n); H_p.append(p); H_th.append(th)
    if not H_n:
        return _empty_axes()
    H_n, H_p, H_th = torch.cat(H_n), torch.cat(H_p), torch.cat(H_th)
    inlier_matrix = torch.cat(H_inl, dim=0)

    avail_pairs = torch.ones(N, dtype=torch.bool, device=dev)
    avail_hyp = torch.ones(H_n.shape[0], dtype=torch.bool, device=dev)
    axes, pts, angs, sdes = [], [], [], []

    for _ in range(int(cfg.max_steps)):
        if len(axes) >= int(cfg.max_axes):
            break
        n_pool = int(avail_pairs.sum().item())
        if n_pool < seed or not avail_hyp.any():
            break
        # candidates = top supported hypotheses in the remaining pool, above the minimum support
        counts = inlier_matrix[:, avail_pairs].sum(dim=1)
        counts = torch.where(avail_hyp, counts, torch.full_like(counts, -1))
        top_cnt, top_idx = torch.topk(counts, min(int(cfg.candidates_per_step), int(avail_hyp.sum().item())))
        sel = top_idx[top_cnt >= min_support(n_pool, cfg.min_support_pct, seed)]
        if sel.numel() == 0:
            break
        P, Q = P_all[avail_pairs], Q_all[avail_pairs]

        # (3b) 1st refinement -> (3a) verification: non-identity, SDE, inlier ratio
        n1, p1, th1, inl1 = refine_rotation(P, Q, H_n[sel], H_p[sel], H_th[sel], thr=cfg.inlier_dist, iters=cfg.refine_iters)
        R1 = rodrigues(n1, th1)
        gate = rotation_gate(th1, inl1, rotation_sde(X, R1, translation(R1, p1)), n_pool, cfg.identity_theta, cfg.sde, cfg.inlier_ratio_pct)

        # (3b) 2nd refinement
        keep = torch.zeros_like(gate)
        if gate.any():
            n2, p2, th2, _ = refine_rotation(P, Q, n1[gate], p1[gate], th1[gate], thr=cfg.inlier_dist, iters=cfg.refine_iters)
            k2 = th2.abs() > cfg.identity_theta
            if k2.any():
                R2 = rodrigues(n2[k2], th2[k2])
                surv_sde = rotation_sde(X, R2, translation(R2, p2[k2]))
                g_idx = torch.nonzero(gate, as_tuple=False).squeeze(1)
                keep[g_idx[torch.nonzero(k2, as_tuple=False).squeeze(1)]] = True
                surv_n, surv_p, surv_th = n2[k2], p2[k2], th2[k2]

        avail_hyp[sel[~keep]] = False                       # failed candidates are consumed (pairs stay)
        if not keep.any():
            continue

        # (4) select (min SDE) + remove support
        b = select_rotation(surv_sde)
        n_b, p_b, th_b = surv_n[b], surv_p[b], surv_th[b]
        axes.append(n_b.cpu()); pts.append(p_b.cpu()); angs.append(th_b.cpu()); sdes.append(surv_sde[b].cpu())
        avail_hyp[sel[torch.nonzero(keep, as_tuple=False).squeeze(1)[b]]] = False
        rm = rotation_removal_mask(P_all, Q_all, n_b, p_b, avail_pairs, cfg.remove_tau)
        if int(rm.sum().item()) == 0:
            break
        avail_pairs[rm] = False

    if not axes:
        return _empty_axes()
    return (torch.stack(axes).numpy().astype(np.float32), torch.stack(angs).numpy().astype(np.float32),
            torch.stack(sdes).numpy().astype(np.float32), torch.stack(pts).numpy().astype(np.float32))
