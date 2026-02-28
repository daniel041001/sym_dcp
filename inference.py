"""
Unified DCP symmetry inference entrypoint.

One script supports both:
- rotation symmetry evaluation (same behavior as dcp_inference_rot.py)
- reflection symmetry evaluation (same behavior as dcp_inference_refl.py)
"""

import argparse
import os.path as osp
import pickle

import torch

from data.preprocessed_reflection_data import load_preprocessed_dataset
from data.symmetrynet_shapenet_gt import build_shapenet_rot_cache
from symmetry_detection.utils.utils import resolve_mesh_root


def _default_model_path(script_dir: str) -> str:
    pretrained_dir = osp.join(script_dir, "pretrained")
    if osp.exists(pretrained_dir):
        if osp.exists(osp.join(pretrained_dir, "dcp_v2.t7")):
            return osp.join(pretrained_dir, "dcp_v2.t7")
        if osp.exists(osp.join(pretrained_dir, "dcp_v1.t7")):
            return osp.join(pretrained_dir, "dcp_v1.t7")
    return None


def _build_rotation_dataset(args, script_dir: str, repo_root: str):
    if args.synthetic:
        print(
            f"Generating toy dataset: n-gonal pyramids (3/4/5/6) x {args.per_class} each "
            f"= {4 * args.per_class} samples, {args.num_points} points each..."
        )
        from data.generate_rotation_synthetic_data import generate_toy_dataset

        pcs, labels, meta_n = generate_toy_dataset(
            per_class=args.per_class,
            num_points=args.num_points,
            seed=args.seed,
            n_sides_list=[3, 4, 5, 6],
            apex_height=args.apex_height,
            trans_range=args.trans_range,
        )
        print(
            f"Generated {len(pcs)} point clouds. Class counts: "
            f"3:{sum(n == 3 for n in meta_n)}, 4:{sum(n == 4 for n in meta_n)}, "
            f"5:{sum(n == 5 for n in meta_n)}, 6:{sum(n == 6 for n in meta_n)}"
        )

        sample_path = osp.join(script_dir, f"pyramid_n{meta_n[0]}_sample.ply")
        with open(sample_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {pcs[0].shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in pcs[0]:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        print(f"Saved sample for viewing: {sample_path}")
        return pcs, labels

    cache_path = args.cache_file or osp.join(script_dir, "data", "shapenet_rot_gt_cache.pkl")
    if osp.isfile(cache_path):
        print(f"Loading cache: {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        pcs = data["point_clouds"]
        labels = data["gt_labels"]
        if args.max_samples is not None and len(pcs) > args.max_samples:
            pcs = pcs[: args.max_samples]
            labels = labels[: args.max_samples]
        print(f"Loaded {len(pcs)} samples from cache")
        return pcs, labels

    dataset_root = args.dataset_root or osp.join(script_dir, "model_symmetry_extracted")
    if not osp.isdir(dataset_root) or not osp.isdir(osp.join(dataset_root, "model_symmetry")):
        print(f"Error: dataset_root (model_symmetry) not found: {dataset_root}")
        print("  Extract model_symmetry.zip to dcp/model_symmetry_extracted/ or set --dataset_root.")
        raise SystemExit(1)

    mesh_root = resolve_mesh_root(args.mesh_root, script_dir, repo_root)
    if not mesh_root:
        print("Error: ShapeNet mesh root not found. Set --mesh_root or put ShapeNetCore.v2 in repo/dcp/.")
        raise SystemExit(1)

    print("Cache not found. Building from model_symmetry + ShapeNetCore.v2...")
    try:
        pcs, labels = build_shapenet_rot_cache(
            dataset_root,
            mesh_root,
            cache_path,
            num_points=args.num_points,
            max_samples=args.max_samples,
        )
    except Exception as e:
        print(f"Error building cache: {e}")
        raise SystemExit(1)

    if args.max_samples is not None and len(pcs) > args.max_samples:
        pcs = pcs[: args.max_samples]
        labels = labels[: args.max_samples]
    return pcs, labels


def _run_rotation(args, script_dir: str, repo_root: str):
    from model.DCP_rotation import DCPInference
    from symmetry_detection.evaluation_rotation import evaluate_rotation_symmetry_by_correspondence

    infer = DCPInference(
        args.model_path,
        device=args.device,
        emb_nn=args.emb_nn,
        pointer=args.pointer,
        head=args.head,
        emb_dims=args.emb_dims,
        topk=args.topk,
        removal_radius=args.removal_radius,
    )
    pcs, labels = _build_rotation_dataset(args, script_dir, repo_root)

    if len(pcs) == 0:
        print("Error: No samples in cache or built. Check dataset_root and mesh_root.")
        raise SystemExit(1)

    precision, recall, f1, avg_ch = evaluate_rotation_symmetry_by_correspondence(
        pcs,
        labels,
        infer,
        dist_threshold=args.dist_threshold,
        chamfer_threshold=args.chamfer_threshold,
        count_entire=args.count_entire,
    )
    print("\n=== Evaluation Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Chamfer:   {avg_ch:.6f}")
    print(f"\nEvaluated {len(pcs)} samples with rotation symmetry GT")


def _run_reflection(args, script_dir: str):
    from model.DCP_reflection import DCPInference
    from symmetry_detection.evaluation_reflection import evaluate_reflection_symmetry_by_correspondence

    # Keep behavior identical to dcp_inference_refl.py.
    infer = DCPInference(
        args.model_path,
        device=args.device,
        emb_nn=args.emb_nn,
        pointer=args.pointer,
        head=args.head,
        emb_dims=args.emb_dims,
    )
    data_root = args.data_root or osp.join(script_dir, "data", "preprocessed_data_r")
    pcs, labels = load_preprocessed_dataset(data_root)

    precision, recall, f1, avg_ch = evaluate_reflection_symmetry_by_correspondence(
        pcs,
        labels,
        [],
        infer,
        dist_threshold=args.dist_threshold,
        chamfer_threshold=args.chamfer_threshold,
        count_entire=args.count_entire,
        refinement_criteria=args.refinement_criteria,
        available_inlier_threshold=args.available_inlier_threshold,
    )
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Chamfer:   {avg_ch:.6f}")


def main():
    script_dir = osp.dirname(osp.abspath(__file__))
    repo_root = osp.abspath(osp.join(script_dir, ".."))

    parser = argparse.ArgumentParser(description="Unified DCP Symmetry Evaluation")
    parser.add_argument(
        "--task",
        type=str,
        default="reflection",
        choices=["reflection", "rotation"],
        help="Choose symmetry task: refl (reflection) or rot (rotation)",
    )

    default_model = _default_model_path(script_dir)
    parser.add_argument(
        "--model_path",
        type=str,
        default=default_model,
        help=f"Path to DCP model weights (default: {default_model if default_model else 'required'})",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--dist_threshold", type=float, default=0.05, help="Distance threshold for matching")
    parser.add_argument("--chamfer_threshold", type=float, default=0.0005, help="Chamfer distance threshold")
    parser.add_argument(
        "--emb_nn",
        type=str,
        default="dgcnn",
        choices=["pointnet", "dgcnn"],
        help="Embedding network type",
    )
    parser.add_argument(
        "--pointer",
        type=str,
        default="transformer",
        choices=["identity", "transformer"],
        help="Pointer type",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="svd",
        choices=["mlp", "svd"],
        help="Head type",
    )
    parser.add_argument("--emb_dims", type=int, default=512, help="Embedding dimensions")

    # Reflection-specific args.
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="(refl) Path to preprocessed reflection dataset (default: data/preprocessed_data_r)",
    )
    parser.add_argument(
        "--count_entire",
        action="store_true",
        help="(refl) Count all point-pairs for inlier scoring",
    )
    parser.add_argument(
        "--refinement_criteria",
        type=float,
        default=10.0,
        help="(refl) Iterative RANSAC refinement criteria (%)",
    )
    parser.add_argument(
        "--available_inlier_threshold",
        type=int,
        default=7,
        help="(refl) Minimum inliers for a plane to be considered in RANSAC (default: 10, set to 0 to disable)",
    )

    # Rotation-specific args.
    parser.add_argument(
        "--topk",
        type=int,
        default=40,
        help="(rot) Top-k correspondences from DCP to consider for RANSAC",
    )
    parser.add_argument(
        "--removal_radius",
        type=float,
        default=0.1,
        help="(rot) Remove identity-like correspondences below this radius (0 to disable)",
    )
    parser.add_argument("--num_points", type=int, default=2048, help="(rot) Number of points per point cloud")
    parser.add_argument("--synthetic", action="store_true", help="(rot) Use synthetic dataset instead of ShapeNet")
    parser.add_argument("--per_class", type=int, default=50, help="(rot) Samples per synthetic shape")
    parser.add_argument("--seed", type=int, default=42, help="(rot) Random seed for synthetic generation")
    parser.add_argument("--apex_height", type=float, default=1.5, help="(rot) Apex height of pyramids")
    parser.add_argument("--trans_range", type=float, default=0.5, help="(rot) Translation range for random SE(3)")
    parser.add_argument(
        "--cache_file",
        type=str,
        default=None,
        help="(rot) Pre-built pkl: point_clouds + gt_labels (default: data/shapenet_rot_gt_cache.pkl)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="(rot) Path containing model_symmetry (used when cache is missing)",
    )
    parser.add_argument(
        "--mesh_root",
        type=str,
        default=None,
        help="(rot) ShapeNetCore.v2 root (used when cache is missing)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="(rot) Maximum samples for cache build/evaluation",
    )

    args = parser.parse_args()

    if args.model_path is None:
        parser.error("--model_path is required. Available models: pretrained/dcp_v1.t7, pretrained/dcp_v2.t7")

    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if args.task == "rotation":
        _run_rotation(args, script_dir, repo_root)
    elif args.task == "reflection":
        if args.topk == 40:
            args.topk = 60
        _run_reflection(args, script_dir)
    else:
        print(f"Unsupported task: {args.task}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
