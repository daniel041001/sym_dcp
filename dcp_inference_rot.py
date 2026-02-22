"""
DCP 모델을 사용하여 correspondence를 추출하는 inference 함수
Rotation symmetry detection을 위한 토이 데이터셋 사용
 
요청 반영(이번 수정):
- 정삼각뿔/정사각뿔/정오각뿔/정육각뿔 4종 데이터셋 생성
- 각 물체당 50개씩 총 200개 샘플 생성
- 각 샘플마다 랜덤 SE(3) (회전 + 평행이동) 적용해서 hard case 구성
- 각 샘플에 대해 rotation symmetry GT(월드 좌표 axis + 모든 비자명 각도 2πk/n, k=1..n-1) 생성
"""
 
import os.path as osp
import argparse

import torch


def main():
    # Optional perf toggles (safe)
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
 
    parser = argparse.ArgumentParser(description='DCP Rotation Symmetry Evaluation (GPU RANSAC)')
    default_model = None
    pretrained_dir = osp.join(osp.dirname(__file__), 'pretrained')
    if osp.exists(pretrained_dir):
        if osp.exists(osp.join(pretrained_dir, 'dcp_v2.t7')):
            default_model = osp.join(pretrained_dir, 'dcp_v2.t7')
        elif osp.exists(osp.join(pretrained_dir, 'dcp_v1.t7')):
            default_model = osp.join(pretrained_dir, 'dcp_v1.t7')
 
    parser.add_argument('--model_path', type=str, default=default_model,
                        help=f'Path to DCP model weights (default: {default_model if default_model else "required"})')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points per point cloud')
    parser.add_argument('--dist_threshold', type=float, default=0.1, help='Distance threshold for matching')
    parser.add_argument('--chamfer_threshold', type=float, default=0.005, help='Chamfer distance threshold')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Embedding network type')
    parser.add_argument('--pointer', type=str, default='transformer', choices=['identity', 'transformer'], help='Pointer type')
    parser.add_argument('--head', type=str, default='svd', choices=['mlp', 'svd'], help='Head type')
    parser.add_argument('--emb_dims', type=int, default=512, help='Embedding dimensions')
 
    # Dataset params
    parser.add_argument('--per_class', type=int, default=50, help='Samples per shape (default 50 -> total 200)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset generation')
    parser.add_argument('--apex_height', type=float, default=1.5, help='Apex height of pyramids')
    parser.add_argument('--trans_range', type=float, default=0.5, help='Translation range for random SE(3)')
 
    args = parser.parse_args()
 
    if args.model_path is None:
        parser.error("--model_path is required. Available models: pretrained/dcp_v1.t7, pretrained/dcp_v2.t7")
 
    from model.DCP import DCPInference
    infer = DCPInference(
        args.model_path,
        device=args.device,
        emb_nn=args.emb_nn,
        pointer=args.pointer,
        head=args.head,
        emb_dims=args.emb_dims
    )
 
    print(f"Generating toy dataset: n-gonal pyramids (3/4/5/6) x {args.per_class} each "
          f"= {4*args.per_class} samples, {args.num_points} points each...")
    from data.generate_rotation_synthetic_data import generate_toy_dataset
    pcs, labels, meta_n = generate_toy_dataset(
        per_class=args.per_class,
        num_points=args.num_points,
        seed=args.seed,
        n_sides_list=[3, 4, 5, 6],
        apex_height=args.apex_height,
        trans_range=args.trans_range,
    )
    print(f"Generated {len(pcs)} point clouds. Class counts: "
          f"3:{sum(n==3 for n in meta_n)}, 4:{sum(n==4 for n in meta_n)}, "
          f"5:{sum(n==5 for n in meta_n)}, 6:{sum(n==6 for n in meta_n)}")
 
    # Save one sample
    sample_path = osp.join(osp.dirname(__file__), f'pyramid_n{meta_n[0]}_sample.ply')
    with open(sample_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pcs[0].shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pcs[0]:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"Saved sample for viewing: {sample_path}")
 
    from symmetry_detection.evaluation_rotation import evaluate_rotation_symmetry_by_correspondence
    precision, recall, f1, avg_ch = evaluate_rotation_symmetry_by_correspondence(
        pcs, labels, infer,
        dist_threshold=args.dist_threshold,
        chamfer_threshold=args.chamfer_threshold
    )
 
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Chamfer:   {avg_ch:.6f}")
 
 
if __name__ == '__main__':
    main()
