#!/usr/bin/env bash
# One-time setup after `git clone`: fetch PARE-Net, build its CUDA extensions, install Python deps.
# Tested with: python 3.8, torch 2.0.1+cu117, CUDA 11.7, pytorch3d 0.7.5.
#   bash setup_symdet.sh            # build + pip deps
#   SKIP_PIP=1 bash setup_symdet.sh # only build PARE-Net
#   SKIP_POINTOPS=1                 # pointops_cuda already installed in this env
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] PARE-Net submodule"
git submodule update --init parenet

echo "[2/4] pointops device patch (launch kernels on the input tensor's GPU)"
if git -C parenet apply --check ../patches/parenet_pointops_device.patch 2>/dev/null; then
    git -C parenet apply ../patches/parenet_pointops_device.patch
else
    echo "      already applied"
fi

if [ -z "${SKIP_PIP:-}" ]; then
    echo "[3/4] python deps (torch / pytorch3d must already match your CUDA; see README.md)"
    pip install -q numpy scipy tqdm easydict einops coloredlogs
fi

echo "[4/4] build PARE-Net extensions"
(cd parenet && python setup.py build_ext --inplace)
if [ -z "${SKIP_POINTOPS:-}" ]; then
    (cd parenet/pareconv/extensions/pointops && python setup.py install)   # installs pointops_cuda into the env
fi

python - <<'EOF'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "parenet"))
import torch  # noqa: F401  (loads libc10 before the extensions)
import pareconv.ext, pointops_cuda  # noqa: F401
print("OK: PARE-Net extensions import")
EOF
echo "Done. Try:  python run_symdet.py --task ref --stride 50"
