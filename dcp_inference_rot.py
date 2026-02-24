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
import sys
import argparse

import torch

# SymmetryNet ShapeNet GT 로더 (원본 메시 → 포인트클라우드는 trimesh)
from data.symmetrynet_shapenet_gt import build_shapenet_rot_cache

def main():
    # Optional perf toggles (safe)
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    script_dir = osp.dirname(osp.abspath(__file__))
    repo_root = osp.abspath(osp.join(script_dir, '..'))
 
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
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points per point cloud')
    parser.add_argument('--dist_threshold', type=float, default=0.1, help='Distance threshold for matching')
    parser.add_argument('--chamfer_threshold', type=float, default=0.005, help='Chamfer distance threshold')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Embedding network type')
    parser.add_argument('--pointer', type=str, default='transformer', choices=['identity', 'transformer'], help='Pointer type')
    parser.add_argument('--head', type=str, default='svd', choices=['mlp', 'svd'], help='Head type')
    parser.add_argument('--emb_dims', type=int, default=512, help='Embedding dimensions')

    # Hyperparamters for RANSAC
    parser.add_argument('--topk', type=int, default=40, help='Top-k correspondences from DCP to consider for RANSAC')
 
    # Synthetic dataset
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic dataset instead of ShapeNet')

    # ShapeNet: 캐시 파일 있으면 그대로 로드, 없으면 dataset_root + mesh_root 로 빌드 후 저장
    parser.add_argument('--cache_file', type=str, default=None,
                        help='Pre-built pkl (point_clouds + gt_labels). Default: dcp/shapenet_rot_gt_cache.pkl')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='model_symmetry 폴더 있는 경로 (캐시 없을 때만). Default: dcp/model_symmetry_extracted')
    parser.add_argument('--mesh_root', type=str, default=None,
                        help='ShapeNetCore.v2 경로 (캐시 없을 때만). Default: env 또는 repo/dcp 내 ShapeNetCore.v2')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='최대 샘플 수 (None = 전부). 캐시 빌드/평가 둘 다 적용')
 
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
        emb_dims=args.emb_dims,
        topk=args.topk,
    )

    if args.synthetic:
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

    else:
        cache_path = args.cache_file or osp.join(script_dir, 'shapenet_rot_gt_cache.pkl')

        # 캐시 있으면 로드, 없으면 dataset_root + mesh_root 로 빌드 후 사용 (name_list.txt 불필요)
        if osp.isfile(cache_path):
            print(f"Loading cache: {cache_path}")
            import pickle
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            point_clouds = data['point_clouds']
            gt_labels = data['gt_labels']
            if args.max_samples is not None and len(point_clouds) > args.max_samples:
                point_clouds = point_clouds[:args.max_samples]
                gt_labels = gt_labels[:args.max_samples]
            print(f"Loaded {len(point_clouds)} samples from cache")
        else:
            if args.dataset_root is None:
                args.dataset_root = osp.join(script_dir, 'model_symmetry_extracted')
            if not osp.isdir(args.dataset_root) or not osp.isdir(osp.join(args.dataset_root, 'model_symmetry')):
                print(f"Error: dataset_root (model_symmetry) not found: {args.dataset_root}")
                print("  Extract model_symmetry.zip to dcp/model_symmetry_extracted/ or set --dataset_root.")
                sys.exit(1)
            args.mesh_root = _resolve_mesh_root(args.mesh_root, script_dir, repo_root)
            if not args.mesh_root:
                print("Error: ShapeNet mesh root not found. Set --mesh_root or put ShapeNetCore.v2 in repo/dcp/.")
                sys.exit(1)
            print("Cache not found. Building from model_symmetry + ShapeNetCore.v2...")
            try:
                point_clouds, gt_labels = build_shapenet_rot_cache(
                    args.dataset_root,
                    args.mesh_root,
                    cache_path,
                    num_points=args.num_points,
                    max_samples=args.max_samples,
                )
            except Exception as e:
                print(f"Error building cache: {e}")
                sys.exit(1)
            if args.max_samples is not None and len(point_clouds) > args.max_samples:
                point_clouds = point_clouds[:args.max_samples]
                gt_labels = gt_labels[:args.max_samples]
    
    if len(point_clouds) == 0:
        print("Error: No samples in cache or built. Check dataset_root and mesh_root.")
        sys.exit(1)
    breakpoint()
    
 
    from symmetry_detection.evaluation_rotation import evaluate_rotation_symmetry_by_correspondence
    precision, recall, f1, avg_ch = evaluate_rotation_symmetry_by_correspondence(
        pcs, labels, infer,
        dist_threshold=args.dist_threshold,
        chamfer_threshold=args.chamfer_threshold
    )
 
    print(f"\n=== Evaluation Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Chamfer:   {avg_ch:.6f}")
    print(f"\nEvaluated {len(point_clouds)} samples with rotation symmetry GT")
 
 
if __name__ == '__main__':
    main()
