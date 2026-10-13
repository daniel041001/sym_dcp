# symdet — reflection / rotation symmetry detection

Modular implementation of the paper's detector (correspondence → hypothesis → verification/refinement → selection).
Defaults reproduce the confirmed settings:

| task | data | full-set F1 |
|---|---|---|
| reflection | `data/preprocessed_data_r` (1000 shapes) | 0.8584 |
| rotation | `data/shapenet_rot_gt_cache.pkl` (1909 shapes) | 0.9682 |

## Setup

Requirements: Linux, NVIDIA GPU, CUDA toolkit matching your PyTorch, Python 3.8.
Tested with torch 2.0.1+cu117, CUDA 11.7, pytorch3d 0.7.5.

```bash
git clone --recurse-submodules -b refactor/symdet-modular <repo-url> && cd <repo>
# torch + pytorch3d first (they depend on your CUDA), e.g.
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
bash setup_symdet.sh        # PARE-Net submodule + CUDA extensions + small deps
```

Included in the repo: evaluation data (both tasks) and the PARE-Net 3DMatch weights (`weights/parenet_3dmatch.pth.tar`,
from the PARE-Net release). PARE-Net itself is a submodule (`parenet/`, upstream commit `71baada`).

## Run

```bash
python run_symdet.py --task ref                 # full reflection set
python run_symdet.py --task rot --seed 0        # full rotation set (sampling is random; --seed makes it reproducible)
python run_symdet.py --task rot --stride 10     # every 10th shape
python run_symdet.py --task rot --set sde=0.0025 remove_tau=0.1    # override any field of RotationConfig / ReflectionConfig
```

On a multi-GPU machine select the GPU with `CUDA_VISIBLE_DEVICES=1 python run_symdet.py ...`.

## Layout

| file | step |
|---|---|
| `symdet/correspondence.py` | (1) PARE-Net matcher; mirrored-copy matching (ref), self-matching with masking (rot) |
| `symdet/generate_hypothesis.py` | (2) 1-pair plane / 2-pair axis hypotheses, residuals, inlier sets |
| `symdet/verification.py` | (3a) SDE and inlier-ratio gates |
| `symdet/refinement.py` | (3b) re-estimation from inliers |
| `symdet/selection.py` | (4) accept + remove supporting pairs |
| `symdet/pipeline.py` | sequential loops `detect_reflection`, `detect_rotation` |
| `symdet/evaluation.py` | datasets and metrics |
| `symdet/config.py` | default settings |

Outputs were checked to be bit-identical to the original research scripts (40 shapes per task, same random seed for rotation).
