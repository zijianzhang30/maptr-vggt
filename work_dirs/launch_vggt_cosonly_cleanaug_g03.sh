#!/usr/bin/env bash
set -euo pipefail
cd /home/zhangzj26/MapTR
source /home/zhangzj26/miniconda3/etc/profile.d/conda.sh
conda activate maptr
export CUDA_HOME=/usr/local/cuda-11.7
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/zhangzj26/MapTR:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,3
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PORT=29644
exec bash tools/dist_train.sh \
  projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_prelss_maptrsync_official4_cleanaug_cosonly_g03.py \
  2 \
  --work-dir /data/public/zhangzj26/maptr_vggt_cosonly_cleanaug_g03
