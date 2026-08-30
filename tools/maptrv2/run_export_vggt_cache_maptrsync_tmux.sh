#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <gpu_id> <start_idx> <end_idx> <output_dir> <ann_file>"
  exit 1
fi
GPU_ID="$1"
START_IDX="$2"
END_IDX="$3"
OUTPUT_DIR="$4"
ANN_FILE="$5"
cd /home/zhangzj26/MapTR
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "[launch] $(date +%F_%T) gpu=$GPU_ID range=${START_IDX}:${END_IDX}"
/home/zhangzj26/miniconda3/envs/maptr/bin/python -u tools/maptrv2/export_vggt_cache.py   --ann-files "$ANN_FILE"   --output-dir "$OUTPUT_DIR"   --model-path /data1/zhangzj26/vggt_ckpts/model.pt   --preprocess-mode maptr_train_geom   --cache-size 37   --maptr-scale 0.5   --maptr-pad-divisor 32   --max-views-per-forward 1   --device cuda:0   --start-index "$START_IDX"   --end-index "$END_IDX"   --log-interval 20   --empty-cache-per-sample
RC=$?
echo "[exit] rc=$RC $(date +%F_%T) gpu=$GPU_ID range=${START_IDX}:${END_IDX}"
exit $RC
