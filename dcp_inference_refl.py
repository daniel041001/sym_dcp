"""
DCP reflection symmetry evaluation entrypoint.

`dcp_inference_rot.py`와 동일한 구조로,
실행 스크립트는 orchestration만 담당하고 세부 구현은 모듈로 분리한다.
"""

import argparse
import os.path as osp

import torch


def main():
    parser = argparse.ArgumentParser(description="DCP Reflection Symmetry Evaluation")

    default_model = None
    pretrained_dir = osp.join(osp.dirname(__file__), "pretrained")
    if osp.exists(pretrained_dir):
        if osp.exists(osp.join(pretrained_dir, "dcp_v2.t7")):
            default_model = osp.join(pretrained_dir, "dcp_v2.t7")
        elif osp.exists(osp.join(pretrained_dir, "dcp_v1.t7")):
            default_model = osp.join(pretrained_dir, "dcp_v1.t7")

    parser.add_argument(
        "--model_path",
        type=str,
        default=default_model,
        help=f"Path to DCP model weights (default: {default_model if default_model else 'required'})",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to preprocessed reflection dataset (default: data/preprocessed_data_r)",
    )
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
    parser.add_argument(
        "--topk",
        type=int,
        default=40,
        help="Top-k correspondences from DCP to consider",
    )
    parser.add_argument(
        "--removal_radius",
        type=float,
        default=0.1,
        help="Remove near-identity correspondences below this radius (0 to disable)",
    )
    parser.add_argument(
        "--count_entire",
        action="store_true",
        help="Count all point-pairs for inlier scoring instead of only putative matches",
    )
    parser.add_argument(
        "--refinement_criteria",
        type=float,
        default=10.0,
        help="Iterative RANSAC refinement criteria (%)",
    )

    args = parser.parse_args()

    if args.model_path is None:
        parser.error("--model_path is required. Available models: pretrained/dcp_v1.t7, pretrained/dcp_v2.t7")

    from model.DCP_reflection import DCPInference

    infer = DCPInference(
        args.model_path, 
        device=args.device,
        emb_nn=args.emb_nn,
        pointer=args.pointer,
        head=args.head,
        emb_dims=args.emb_dims
    )

    from data.preprocessed_reflection_data import load_preprocessed_dataset

    data_root = args.data_root or osp.join(osp.dirname(__file__), "data", "preprocessed_data_r")
    pcs, labels = load_preprocessed_dataset(data_root)

    from symmetry_detection.evaluation_reflection import evaluate_reflection_symmetry_by_correspondence

    precision, recall, f1, avg_ch = evaluate_reflection_symmetry_by_correspondence(
        pcs,
        labels,
        [],
        infer,
        dist_threshold=args.dist_threshold,
        chamfer_threshold=args.chamfer_threshold,
        count_entire=args.count_entire,
        refinement_criteria=args.refinement_criteria,
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Chamfer:   {avg_ch:.6f}")


if __name__ == "__main__":
    # Optional perf toggles (safe)
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    main()
