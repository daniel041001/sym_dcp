"""
SymmetryNet ShapeNet rotation GT 로더

SymmetryNet ShapeNet 데이터셋의 model_sym.txt에서 회전 대칭 GT를 읽어
dcp 평가 형식(axis 3 + angle 1)으로 변환합니다.

model_sym.txt 형식:
  - 1행: center (3) - 대칭 중심
  - 2~5행: syms (4x3) - 대칭 벡터. 행 인덱스 3만 비영이면 회전 대칭(180°).
  - 회전일 때: axis = syms[3] / ||syms[3]||, angle = 180° (π)
"""
import os
import os.path as osp
import numpy as np
from typing import List, Tuple, Optional

# SymmetryNet dataset_shapenet.py와 동일한 class_id
SHAPENET_CLASS_ID = {
    '000': '02691156', '001': '02747177', '002': '02773838', '003': '02808440',
    '004': '02818832', '005': '02828884', '006': '02876657', '007': '02924116',
    '008': '02942699', '009': '02946921', '010': '02958343', '011': '03001627',
    '012': '03211117', '013': '03261776', '014': '03467517', '015': '03513137',
    '016': '03593526', '017': '03636649', '018': '03642806', '019': '03991062',
    '020': '04225987', '021': '04256520', '022': '04379243',
}

# ShapeNet rotation은 180°
SHAPENET_ROTATION_ANGLE_RAD = np.pi


def load_model_sym_rotation(sym_file: str) -> Tuple[bool, Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """
    SymmetryNet model_sym.txt에서 회전 대칭 GT만 읽습니다.

    Returns
    -------
    has_rotation : bool
        회전 대칭이 있으면 True (row 3만 비영).
    axis : (3,) float32 or None
        단위 축 벡터 (모델 좌표계).
    angle_rad : float or None
        회전 각도(rad). ShapeNet은 180° = π.
    center : (3,) float32 or None
        축 위 한 점(대칭 중심, 모델 좌표계).
    """
    if not osp.isfile(sym_file):
        return False, None, None, None

    model_s = np.loadtxt(sym_file, dtype=np.float64)
    if model_s.size == 0 or model_s.ndim != 2:
        return False, None, None, None

    center = np.asarray(model_s[0, :3], dtype=np.float32)
    syms = model_s[1:5, :3]  # (4, 3)

    # 비영인 행만 추출 (행 인덱스 0,1,2,3)
    check = np.any(syms != 0, axis=1)
    row_id = np.where(check)[0]

    # 회전 대칭: row_id가 [3] 하나일 때만 (인덱스 3 = 4번째 행)
    if row_id.shape[0] != 1 or row_id[0] != 3:
        return False, None, None, None

    axis_vec = np.asarray(syms[3], dtype=np.float32)
    n = np.linalg.norm(axis_vec)
    if n < 1e-8:
        return False, None, None, None

    axis = (axis_vec / n).astype(np.float32)
    return True, axis, SHAPENET_ROTATION_ANGLE_RAD, center


def rotation_gt_to_dcp_format(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """dcp gt_labels 한 개 형식: (4,) = axis(3) + angle(1)."""
    return np.concatenate([np.asarray(axis, dtype=np.float32).ravel()[:3], np.array([angle_rad], dtype=np.float32)])


def get_sym_file_path(dataset_root: str, project_root: str, line: str) -> Tuple[str, str, str]:
    """
    train_ls.txt 한 줄(예: '000/0001/0')에서 sym_file 경로와 class_name, ins_name을 구합니다.
    project_root에는 name_list.txt가 있어야 합니다.
    """
    parts = line.strip().split('/')
    if len(parts) < 2:
        raise ValueError(f"Invalid list line: {line}")
    class_key = parts[0][:3]
    ins_num = int(parts[1]) if len(parts[1]) else 0

    class_name = SHAPENET_CLASS_ID.get(class_key)
    if class_name is None:
        raise ValueError(f"Unknown class_key: {class_key}")

    name_list_path = osp.join(project_root, 'name_list.txt')
    if not osp.isfile(name_list_path):
        raise FileNotFoundError(f"name_list.txt not found: {name_list_path}")

    name_list = np.loadtxt(name_list_path, dtype=str, delimiter='\n')
    cls_idx = int(class_key)
    if cls_idx >= len(name_list):
        raise IndexError(f"class_key {class_key} out of range for name_list")
    instance_ls = name_list[cls_idx][1:-1].split(",")
    ins_name = instance_ls[ins_num][2:-1]

    sym_dir = osp.join(dataset_root, 'model_symmetry')
    sym_file = osp.join(sym_dir, class_name, ins_name, 'model_sym.txt')
    return sym_file, class_name, ins_name


def load_rt(rt_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """4x4 rt 파일에서 R (3,3), t (3,) 반환."""
    rt = np.loadtxt(rt_file)
    R = rt[:3, :3]
    t = rt[:3, 3]
    return R, t


def get_shapenet_rotation_gt(
    dataset_root: str,
    project_root: str,
    split: str = 'train',
    transform_to_camera: bool = False,
    list_file_override: Optional[str] = None,
) -> Tuple[List[dict], List[List[np.ndarray]]]:
    """
    SymmetryNet ShapeNet 스플릿에 대해 회전 GT 목록을 구합니다.

    Parameters
    ----------
    dataset_root : str
        ShapeNet 데이터 루트 (model_symmetry/, rgbd/ 가 있는 경로).
    project_root : str
        SymmetryNet 프로젝트 루트 (name_list.txt, datasets/shapenet/dataset_config/).
    split : str
        'train' | 'holdout_instance' | 'holdout_view' | 'holdout_class'
    transform_to_camera : bool
        True면 각 샘플의 rt로 축을 카메라(뷰) 좌표계로 변환. rt 파일 경로가 sample_infos에 필요.
    list_file_override : str, optional
        지정하면 split 대신 이 파일을 리스트로 사용.

    Returns
    -------
    sample_infos : list of dict
        각 항목: sym_file, class_name, ins_name, view, (rt_file if transform_to_camera)
    gt_labels : list of list of np.ndarray
        dcp 형식. gt_labels[i] = [ np.array([ax,ay,az,angle]), ... ] (회전 없으면 [])
    """
    if list_file_override is not None:
        list_path = list_file_override
    else:
        list_path = osp.join(
            project_root, 'datasets', 'shapenet', 'dataset_config',
            {'train': 'train_ls.txt', 'holdout_instance': 'holdout_ins_ls.txt',
             'holdout_view': 'holdout_view_ls.txt', 'holdout_class': 'holdout_class_ls.txt'}[split]
        )

    if not osp.isfile(list_path):
        raise FileNotFoundError(f"List file not found: {list_path}")

    with open(list_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    sample_infos = []
    gt_labels = []

    for line in lines:
        parts = line.split('/')
        view = parts[2] if len(parts) > 2 else '0'

        try:
            sym_file, class_name, ins_name = get_sym_file_path(dataset_root, project_root, line)
        except (ValueError, FileNotFoundError, IndexError) as e:
            continue

        info = {'sym_file': sym_file, 'class_name': class_name, 'ins_name': ins_name, 'view': view, 'line': line}

        if transform_to_camera:
            # rgbd/{split}/{line}-rt.txt
            rt_file = osp.join(dataset_root, 'rgbd', split, f"{line}-rt.txt")
            info['rt_file'] = rt_file

        has_rot, axis, angle_rad, center = load_model_sym_rotation(sym_file)
        if not has_rot:
            sample_infos.append(info)
            gt_labels.append([])
            continue

        axis_model = axis
        if transform_to_camera and info.get('rt_file') and osp.isfile(info['rt_file']):
            R, t = load_rt(info['rt_file'])
            # 방향만 회전 (축은 방향 벡터)
            axis_cam = (R @ axis_model).astype(np.float32)
            n = np.linalg.norm(axis_cam)
            if n > 1e-8:
                axis_cam = (axis_cam / n).astype(np.float32)
            axis_model = axis_cam

        param = rotation_gt_to_dcp_format(axis_model, angle_rad)
        sample_infos.append(info)
        gt_labels.append([param])

    return sample_infos, gt_labels


def build_shapenet_rot_cache(
    dataset_root: str,
    mesh_root: str,
    output_path: str,
    num_points: int = 2048,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """
    model_symmetry 폴더만 스캔해서 회전 GT 있는 것만 골라, 메시에서 PC 샘플링 후 캐시 저장.
    name_list.txt / dataset_config 불필요.

    Returns
    -------
    point_clouds : (N, M, 3) float32
    gt_labels : list of list of (4,) float32
    """
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, *a, **k):
            return x

    sym_base = osp.join(dataset_root, 'model_symmetry')
    if not osp.isdir(sym_base):
        raise FileNotFoundError(f"model_symmetry not found: {sym_base}")

    pairs: List[Tuple[np.ndarray, List[np.ndarray]]] = []
    for class_id in sorted(os.listdir(sym_base)):
        class_dir = osp.join(sym_base, class_id)
        if not osp.isdir(class_dir):
            continue
        for ins_id in tqdm(sorted(os.listdir(class_dir)), desc=class_id, leave=False):
            sym_file = osp.join(class_dir, ins_id, 'model_sym.txt')
            if not osp.isfile(sym_file):
                continue
            has_rot, axis, angle_rad, _ = load_model_sym_rotation(sym_file)
            if not has_rot:
                continue
            mesh_path = osp.join(mesh_root, class_id, ins_id, 'models', 'model_normalized.obj')
            if not osp.isfile(mesh_path):
                continue
            pc = load_point_cloud_from_mesh(mesh_path, num_points=num_points)
            if pc is None or pc.shape[0] < 100:
                continue
            param = rotation_gt_to_dcp_format(axis, angle_rad)
            pairs.append((pc, [param]))
            if max_samples is not None and len(pairs) >= max_samples:
                break
        if max_samples is not None and len(pairs) >= max_samples:
            break

    if not pairs:
        raise RuntimeError("No (mesh, rotation GT) pairs found. Check dataset_root and mesh_root.")

    point_clouds = np.array([p[0] for p in pairs], dtype=np.float32)
    gt_labels = [p[1] for p in pairs]

    import pickle
    out_dir = osp.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({'point_clouds': point_clouds, 'gt_labels': gt_labels}, f)
    print(f"Saved cache: {output_path} ({len(point_clouds)} samples)")
    return point_clouds, gt_labels


def load_point_cloud_from_mesh(
    mesh_path: str,
    num_points: int = 2048,
    dense_sample: int = 40000,
) -> Optional[np.ndarray]:
    """
    OBJ 메시에서 포인트 클라우드 샘플링 (trimesh 사용, data_procesor 미사용).

    Parameters
    ----------
    mesh_path : str
        .obj 메시 파일 경로
    num_points : int
        목표 포인트 수 (기본 2048)
    dense_sample : int
        표면 밀집 샘플 수 (이 중에서 num_points까지 다운샘플)

    Returns
    -------
    (N, 3) float32 or None
        중심 정규화된 포인트 클라우드. 실패 시 None.
    """
    if not osp.isfile(mesh_path):
        return None
    try:
        import trimesh
    except ImportError:
        raise ImportError("load_point_cloud_from_mesh requires trimesh. Install with: pip install trimesh")
    try:
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        n_sample = min(dense_sample, num_points * 20)
        n_sample = max(n_sample, num_points)
        points, _ = trimesh.sample.sample_surface(mesh, n_sample)
        points = np.asarray(points, dtype=np.float32)
        if points.shape[0] > num_points:
            idx = np.random.choice(points.shape[0], num_points, replace=False)
            points = points[idx]
        elif points.shape[0] < num_points:
            idx = np.random.choice(points.shape[0], num_points - points.shape[0], replace=True)
            points = np.concatenate([points, points[idx]], axis=0)
        points = points - points.mean(axis=0, keepdims=True)
        return points
    except Exception:
        return None


def load_point_cloud_from_symmetrynet_rgbd(
    dataset_root: str,
    split: str,
    line: str,
    num_points: int = 2048,
    cam_scale: float = 100.0,
) -> np.ndarray:
    """
    SymmetryNet RGB-D에서 해당 뷰의 포인트 클라우드를 생성합니다.
    (선택) dcp에서 2048점 쓰면 여기서 2048로 맞출 수 있음.
    """
    import cv2
    import numpy.ma as ma

    root = osp.join(dataset_root, 'rgbd', split)
    depth = cv2.imread(osp.join(root, f"{line}-depth-crop-occlusion.png"))
    if depth is None:
        return np.zeros((num_points, 3), dtype=np.float32)
    depth = depth[:, :, 0]
    cam = np.loadtxt(osp.join(root, f"{line}-k-crop.txt"))
    cam_cx, cam_cy = cam[0, 2], cam[1, 2]
    cam_fx, cam_fy = cam[0, 0], cam[1, 1]

    h, w = depth.shape
    xmap = np.array([[i for i in range(w)] for _ in range(h)], dtype=np.float32)
    ymap = np.array([[j for _ in range(w)] for j in range(h)], dtype=np.float32)

    mask = ma.getmaskarray(ma.masked_not_equal(depth, 255))
    choose = mask.flatten().nonzero()[0]
    if len(choose) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    if len(choose) >= num_points:
        np.random.shuffle(choose)
        choose = choose[:num_points]
    else:
        choose = np.pad(choose, (0, num_points - len(choose)), mode='wrap')

    depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = depth_masked / cam_scale
    pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
    cloud = np.concatenate([pt0, pt1, pt2], axis=1)
    return cloud.astype(np.float32)


# ---------------------------------------------------------------------------
# 사용 예 (dcp 평가에 ShapeNet rotation GT 쓰기)
# ---------------------------------------------------------------------------
#
# 1) 디렉터리 구조 (SymmetryNet ShapeNet):
#    dataset_root/
#      model_symmetry/<class_id>/<ins_name>/model_sym.txt
#      rgbd/train/, rgbd/holdout_instance/ ... (이미지, *-rt.txt, *-k-crop.txt)
#    project_root/  (SymmetryNet repo 루트)
#      name_list.txt
#      datasets/shapenet/dataset_config/train_ls.txt, holdout_ins_ls.txt, ...
#
# 2) 모델 좌표계 GT만 필요할 때 (포인트 클라우드는 별도 샘플링):
#    from dcp.symmetrynet_shapenet_gt import get_shapenet_rotation_gt
#    sample_infos, gt_labels = get_shapenet_rotation_gt(
#        dataset_root='/path/to/shapenet',
#        project_root='/path/to/SymmetryNet',
#        split='holdout_instance',
#        transform_to_camera=False,
#    )
#    # 회전 GT 있는 샘플만 필터
#    indices = [i for i, g in enumerate(gt_labels) if len(g) > 0]
#    gt_labels = [gt_labels[i] for i in indices]
#    # 포인트 클라우드는 직접 로드 후 point_clouds, gt_labels를 dcp evaluate에 전달.
#
# 3) 카메라(뷰) 좌표계 GT + 동일 뷰 포인트 클라우드 (dcp 평가와 좌표계 일치):
#    sample_infos, gt_labels = get_shapenet_rotation_gt(
#        dataset_root, project_root, split='holdout_instance',
#        transform_to_camera=True,
#    )
#    point_clouds = []
#    valid_gt = []
#    for i, info in enumerate(sample_infos):
#        if len(gt_labels[i]) == 0:
#            continue
#        pc = load_point_cloud_from_symmetrynet_rgbd(
#            dataset_root, split, info['line'], num_points=2048,
#        )
#        point_clouds.append(pc)
#        valid_gt.append(gt_labels[i])
#    # evaluate_rotation_symmetry_by_correspondence(point_clouds, valid_gt, infer, ...)

    depth = depth[:, :, 0]
    cam = np.loadtxt(osp.join(root, f"{line}-k-crop.txt"))
    cam_cx, cam_cy = cam[0, 2], cam[1, 2]
    cam_fx, cam_fy = cam[0, 0], cam[1, 1]
