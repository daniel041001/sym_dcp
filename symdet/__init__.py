"""Modular reflection / rotation symmetry detection (confirmed 2026-10-13 settings).

    correspondence.py       (1) PARE-Net matcher, reflection / rotation correspondences
    generate_hypothesis.py  (2) minimal-sample hypotheses and residuals / inlier sets
    verification.py         (3a) SDE and inlier-ratio gates
    refinement.py           (3b) re-estimation from inliers
    selection.py            (4) accept and remove support
    pipeline.py             sequential loops (Algorithm 1)
    evaluation.py           datasets and metrics (unchanged from the originals)
    config.py               defaults = confirmed settings
"""
from .config import MatcherConfig, ReflectionConfig, RotationConfig
from .pipeline import detect_reflection, detect_rotation
