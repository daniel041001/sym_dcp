# 3D Symmetry Analysis

## Prerequisites 
PyTorch>=1.0: https://pytorch.org

scipy>=1.2 

numpy

h5py

tqdm

TensorboardX: https://github.com/lanpa/tensorboardX

## Environment Setting
```
conda create -n symmetry python=3.10 -y
conda activate symmetry

export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install scipy numpy h5py tqdm -y
pip install tensorboardX
python -c "import torch, scipy, numpy, h5py, tqdm, tensorboardX; print('OK')"

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e . --no-build-isolation
```

## Inference

### Reflection
```
python dcp_inference.py --count_entire --device cuda:0 --refinement_criteria 10
```

### Rotation
```
python dcp_inference_rot.py
```