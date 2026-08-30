#!/usr/bin/env bash
set -euo pipefail
CONFIG="$1"; OUT="$2"; shift 2
python tools/analysis_tools/gradient_cosine_analysis.py "$CONFIG" --checkpoints "$@" --batches "${NUM_BATCHES:-100}" --groups img_backbone_layer4 --objectives loss_vggt_img_feat --out-dir "$OUT"
python tools/analysis_tools/plot_gradient_conflict.py "$OUT/summary.csv" --out-dir "$OUT"
