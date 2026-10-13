"""Default settings. Running with these reproduces the reported results:
reflection full F1 0.8584 (1000 shapes), rotation full F1 0.9682 (1909 shapes)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatcherConfig:
    weights: str = "weights/parenet_3dmatch.pth.tar"   # relative to the repo root
    device: str = "cuda:0"
    init_voxel_size: float = 0.0125
    num_correspondences: int = 8000                    # PARE-Net fine_matching.num_hypotheses


@dataclass
class ReflectionConfig:
    mirror_axis: int = 2              # mirrored copy: reflect across the plane z = 0
    inlier_dist: float = 0.02         # residual threshold for inliers
    pool_sde: float = 0.001           # SDE threshold on raw hypotheses (pool pruning)
    sde: float = 0.00025              # SDE threshold after refinement
    min_support_pct: float = 0.1      # stop when the best hypothesis explains < this % of all pairs
    inlier_ratio_pct: float = 5.0     # refined inliers must be >= this % of the remaining pool
    refine_iters: int = 1             # iterations per refinement pass (two passes)
    max_steps: int = 2000


@dataclass
class RotationConfig:
    self_mask_radius: float = 0.15    # mask near-identity matches inside the matcher
    min_radius: float = 0.1           # drop pairs with ||x - x'|| < min_radius
    num_hypotheses: int = 40000
    batch: int = 256
    inlier_dist: float = 0.02
    pool_sde: float = 0.004
    sde: float = 0.001
    identity_theta: float = 0.15      # reject |theta| <= this (rad)
    min_support_pct: float = 0.1      # of the remaining pool
    inlier_ratio_pct: float = 0.2
    refine_iters: int = 1
    candidates_per_step: int = 64
    remove_tau: float = 0.3           # remove pairs with sqrt(dr^2 + dh^2) < tau around the accepted axis
    max_axes: int = 8
    max_steps: int = 200
    eval_identity_theta: float = 1e-2  # evaluation ignores near-identity outputs
