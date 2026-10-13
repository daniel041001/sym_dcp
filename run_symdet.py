"""Evaluate reflection or rotation symmetry detection with the modular `symdet` package.

    python run_symdet.py --task ref                 # reflection, confirmed settings (full F1 0.8584)
    python run_symdet.py --task rot                 # rotation,   confirmed settings (full F1 0.9682)
    python run_symdet.py --task rot --stride 10 --seed 0 --set sde=0.0025 remove_tau=0.1
Per-prediction dumps go to symdet_{task}_{tag}.txt (same columns as the original scripts).
"""
from __future__ import annotations

import argparse
import dataclasses
import os.path as osp
import time

import numpy as np
import torch

from symdet import MatcherConfig, ReflectionConfig, RotationConfig, detect_reflection, detect_rotation
from symdet.correspondence import PareNetMatcher, mirror_matrix, reflection_correspondences, rotation_correspondences
from symdet.evaluation import ReflectionScorer, RotationScorer, load_reflection_dataset, load_rotation_dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *a, **k):
        return x

HERE = osp.dirname(osp.abspath(__file__))


def _apply_overrides(cfg, items):
    fields = {f.name: f for f in dataclasses.fields(cfg)}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in fields:
            raise SystemExit(f"unknown setting '{k}' for {type(cfg).__name__}; options: {sorted(fields)}")
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v))
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["ref", "rot"], required=True)
    ap.add_argument("--data", default=None, help="ref: dataset dir (point_clouds.pkl, labels.pkl); rot: gt cache pkl")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=None, help="seed torch per object (rotation sampling is random; default: unseeded, like the original)")
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override ReflectionConfig / RotationConfig fields")
    args = ap.parse_args()

    matcher = PareNetMatcher(MatcherConfig(device=args.device))
    dev = matcher.device
    if args.task == "ref":
        cfg = _apply_overrides(ReflectionConfig(), args.set)
        pcs, labels = load_reflection_dataset(args.data or "data/preprocessed_data_r")
        scorer, M = ReflectionScorer(dev), mirror_matrix(cfg.mirror_axis)
    else:
        cfg = _apply_overrides(RotationConfig(), args.set)
        pcs, labels = load_rotation_dataset(args.data or "data/shapenet_rot_gt_cache.pkl")
        scorer = RotationScorer(dev, cfg.eval_identity_theta)
        matcher.install_self_mask(cfg.self_mask_radius)
    print(cfg)

    idx = list(range(0, len(pcs), max(args.stride, 1)))
    if args.max_samples:
        idx = idx[: args.max_samples]
    t0 = time.time()
    for i in tqdm(idx, dynamic_ncols=True):
        pc = np.asarray(pcs[i], np.float32)
        X = torch.from_numpy(pc).float().to(dev)
        if args.task == "ref":
            x, xp = reflection_correspondences(matcher, pc, M)
            Xc = X.mean(dim=0)
            n, d, _, sde = detect_reflection(x, xp, X - Xc, cfg)
            scorer.add(i, labels[i], n, d, sde, Xc)
        else:
            if len(labels[i]) == 0:
                scorer.add(i, labels[i], np.zeros((0, 3), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros((0, 3), np.float32))
                continue
            x, xp = rotation_correspondences(matcher, pc, cfg.min_radius)
            if args.seed is not None:
                torch.manual_seed(args.seed + i)
            axes, ang, sde, pts = detect_rotation(x, xp, X, cfg)
            scorer.add(i, labels[i], axes, ang, sde, pts)
    el = time.time() - t0

    out = osp.join(HERE, f"symdet_{args.task}_{args.tag}.txt")
    with open(out, "w") as f:
        f.write(scorer.header + "\n" + "\n".join(scorer.lines) + ("\n" if scorer.lines else ""))
    print(f"\nsamples: {len(idx)}   dump: {out}")
    if args.task == "ref":
        P, R, F = scorer.result()
        print(f"Precision: {P:.4f}\nRecall:    {R:.4f}\nF1:        {F:.4f}")
    else:
        P, R, F, avg = scorer.result()
        for thr, p, r, f in zip(scorer.thresholds, P, R, F):
            print(f"dist_threshold = {thr:.2f} -> Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
        print(f"Average F1 over thresholds: {avg:.4f}")
    print(f"Elapsed: {el:.1f}s ({el / 60:.1f} min)")


if __name__ == "__main__":
    main()
