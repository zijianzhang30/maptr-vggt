#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <gpu_id> <start_idx> <end_idx> <session_name> <log_path>"
  exit 1
fi

gpu_id="$1"
start_idx="$2"
end_idx="$3"
session_name="$4"
log_path="$5"

repo_root="/home/zhangzj26/MapTR"
ann_file="/data1/zhangzj26/nuScenes_full/nuscenes_map_infos_temporal_train.pkl"
out_dir="/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_maptrsync_p37"
model_path="/data1/zhangzj26/vggt_ckpts/model.pt"
python_bin="/home/zhangzj26/miniconda3/envs/maptr/bin/python"

mkdir -p "$(dirname "$log_path")"
cd "$repo_root"

export CUDA_VISIBLE_DEVICES="$gpu_id"
{
  echo "[launch] gpu=$gpu_id range=${start_idx}:${end_idx} time=$(date '+%F %T')"
  "$python_bin" -u tools/maptrv2/export_vggt_cache.py \
    --ann-files "$ann_file" \
    --output-dir "$out_dir" \
    --model-path "$model_path" \
    --preprocess-mode maptr_train_geom \
    --cache-size 37 \
    --maptr-scale 0.5 \
    --maptr-pad-divisor 32 \
    --max-views-per-forward 1 \
    --device cuda:0 \
    --start-index "$start_idx" \
    --end-index "$end_idx" \
    --log-interval 20 \
    --empty-cache-per-sample
  rc=$?
  echo "[exit] rc=$rc time=$(date '+%F %T')"
  exit "$rc"
} >> "$log_path" 2>&1
