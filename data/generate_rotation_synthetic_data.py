from typing import List, Tuple

import numpy as np

# -----------------------------
# NEW: multi-shape toy dataset
# -----------------------------
def generate_toy_dataset(
    per_class: int = 50,
    num_points: int = 2048,
    seed: int = 42,
    n_sides_list: List[int] = [3, 4, 5, 6],
    apex_height: float = 1.5,
    trans_range: float = 0.5,
) -> Tuple[np.ndarray, List[List[np.ndarray]], List[int]]:
    """
    Regular n-gonal pyramid dataset (4 classes).
      - n_sides=3: triangular pyramid (정삼각뿔)
      - n_sides=4: square pyramid (정사각뿔)
      - n_sides=5: pentagonal pyramid (정오각뿔)
      - n_sides=6: hexagonal pyramid (정육각뿔)
 
    Each sample: random SE(3) applied (hard case)
    GT: axis in WORLD coords + all non-trivial rotation angles 2πk/n, k=1..n-1
 
    Returns:
      pcs: (total, N, 3)
      labels: list length total, each is list of [axis(3)+angle(1)]
      meta_n: list length total, n_sides per sample
    """
    np.random.seed(seed)
 
    def random_rotation_matrix() -> np.ndarray:
        A = np.random.randn(3, 3).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        return Q.astype(np.float32)
 
    def generate_ngonal_pyramid(n_sides: int, n_points: int) -> np.ndarray:
        # Regular n-gon base on z=0
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False).astype(np.float32) + np.pi / 2
        base_vertices = np.column_stack([
            np.cos(angles),
            np.sin(angles),
            np.zeros(n_sides, dtype=np.float32),
        ]).astype(np.float32)
 
        base_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        apex = np.array([0.0, 0.0, apex_height], dtype=np.float32)
 
        # Sampling budget
        n_base = int(n_points * 0.35)
        n_lateral = n_points - n_base
 
        pts = []
 
        # Base (fan triangulation)
        for _ in range(n_base):
            i = np.random.randint(n_sides)
            v0 = base_center
            v1 = base_vertices[i]
            v2 = base_vertices[(i + 1) % n_sides]
            r1, r2 = np.random.rand(2).astype(np.float32)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            pts.append(r1 * v0 + r2 * v1 + (1 - r1 - r2) * v2)
 
        # Lateral faces
        for _ in range(n_lateral):
            i = np.random.randint(n_sides)
            v0 = apex
            v1 = base_vertices[i]
            v2 = base_vertices[(i + 1) % n_sides]
            r1, r2 = np.random.rand(2).astype(np.float32)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            pts.append(r1 * v0 + r2 * v1 + (1 - r1 - r2) * v2)
 
        pts = np.array(pts, dtype=np.float32)
        pts = pts - pts.mean(axis=0, keepdims=True)
        return pts
 
    pcs: List[np.ndarray] = []
    labels: List[List[np.ndarray]] = []
    meta_n: List[int] = []
 
    for n_sides in n_sides_list:
        for _ in range(per_class):
            base_pc = generate_ngonal_pyramid(n_sides, num_points)
 
            # Hard case: random SE(3)
            Rw = random_rotation_matrix()
            tw = np.random.uniform(-trans_range, trans_range, size=(3,)).astype(np.float32)
            pc_world = (Rw @ base_pc.T).T + tw
 
            pcs.append(pc_world.astype(np.float32))
            meta_n.append(n_sides)
 
            # GT axis in WORLD coords (local +Z rotated by Rw)
            z_local = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            z_world = (Rw @ z_local.reshape(3, 1)).reshape(3).astype(np.float32)
            z_world = z_world / (np.linalg.norm(z_world) + 1e-8)
 
            # GT angles: all non-trivial rotations around axis
            gt_list = []
            for k in range(1, n_sides):
                ang = (2.0 * np.pi * k) / float(n_sides)
                gt_list.append(np.concatenate([z_world, np.array([ang], dtype=np.float32)]))
            labels.append(gt_list)
 
    return np.array(pcs, dtype=np.float32), labels, meta_n
