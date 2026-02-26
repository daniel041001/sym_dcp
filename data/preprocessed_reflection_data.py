import glob
import os.path as osp
import pickle
from typing import List, Tuple

import numpy as np


def load_preprocessed_dataset(dataset_root: str) -> Tuple[np.ndarray, List]:
    """
    Load preprocessed reflection dataset from directory.

    Supports:
    - point_clouds.pkl + labels.pkl
    - per-sample .npy files (optionally *_labels.npy sidecar)
    """
    pcd_file = osp.join(dataset_root, "point_clouds.pkl")
    labels_file = osp.join(dataset_root, "labels.pkl")

    if osp.exists(pcd_file) and osp.exists(labels_file):
        with open(pcd_file, "rb") as f:
            point_clouds = pickle.load(f)
        with open(labels_file, "rb") as f:
            labels = pickle.load(f)
        return point_clouds, labels

    pcd_files = sorted(glob.glob(osp.join(dataset_root, "*.npy")))
    if len(pcd_files) == 0:
        raise FileNotFoundError(f"No data found in {dataset_root}")

    point_clouds = []
    labels = []
    for pcd_path in pcd_files:
        data = np.load(pcd_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            item = data.item()
            point_clouds.append(item["points"])
            labels.append(item.get("labels", []))
        else:
            point_clouds.append(data)
            label_file = pcd_path.replace(".npy", "_labels.npy")
            if osp.exists(label_file):
                label_data = np.load(label_file, allow_pickle=True)
                labels.append(label_data.tolist() if hasattr(label_data, "tolist") else [])
            else:
                labels.append([])

    return np.array(point_clouds), labels
