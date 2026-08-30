# MapTR nuScenes training/eval reproduction log

Last updated: 2026-07-15 (Asia/Shanghai)
Repo: `/home/zhangzj26/MapTR`
Branch observed during work: `maptrv2`
Conda env: `maptr`
Goal: reproduce nuScenes preprocessing, evaluation, and then training for MapTR / MapTRv2.

## Current experiment status | 2026-08-25

### Planned gradient-cosine diagnosis (A1/A2)

To test whether late-stage task mismatch causes VGGT distillation to stop
helping, analyze saved checkpoints on the same fixed training batches. For the
shared image/BEV feature encoder parameters, compute

`g_map = grad(L_Map)` and `g_dist = grad(L_dist)`, where `L_dist` is
`loss_vggt_feat` for A2 (or `loss_vggt_img_feat` for A1), then record
`rho = cosine(g_map, g_dist)` together with both gradient norms. The first
checkpoints are `epoch_4.pth`, `epoch_8.pth`, and `epoch_12.pth`; later epochs
should be added when available. Use `model.eval()`, identical fixed batches,
and report mean/std over 20--50 batches. A positive rho in epochs 4--8 followed
by rho approaching zero or becoming negative in epochs 12--24 would support
the task-mismatch hypothesis. A useful control is to compare a decayed
distillation weight against the fixed-weight run.

Analysis core: `tools/analysis_tools/gradient_cosine_analysis.py`. It must be
called from a runner that builds the MapTR training dataloader and exposes the
two scalar losses; ordinary inference cannot produce these gradients.

- Active/most recent run: `work_dirs/maptrv2_vggt_prelss_maptrsync_official4_cleanaug_aug24`
- Config: `maptrv2_nusc_r50_24ep_vggt_prelss_maptrsync_official4_cleanaug.py`
- Method: pre-LSS VGGT distillation with aligned `maptrsync` cache; clean augmentation (PhotoMetric distortion and GridMask disabled); 4 GPUs, per-GPU batch 4, global batch 16, LR `3e-4` scaled from the official setting.
- Checkpoints available: `epoch_2.pth`, `epoch_4.pth`, `epoch_6.pth`, `epoch_8.pth`, `epoch_10.pth`.
- Latest observed training log: Epoch 11, iter 1100/3517 (2026-08-25 10:43 CST); training loss about 34.66. Epoch 11 validation result was not yet recorded at the time of this update.

### Validation comparison with the official baseline

| Epoch | Run | divider AP | ped_crossing AP | boundary AP | mAP |
|---:|---|---:|---:|---:|---:|
| 8 | VGGT cleanaug | 0.4670 | 0.4520 | 0.5158 | **0.4782** |
| 8 | Official baseline | 0.4504 | 0.4208 | 0.5041 | **0.4584** |
| 10 | VGGT cleanaug | 0.4949 | 0.4748 | 0.5393 | **0.5030** |
| 10 | Official baseline | 0.4533 | 0.4556 | 0.5317 | **0.4802** |

At the same epoch, the VGGT run is ahead by +0.0198 mAP at epoch 8 and +0.0228 mAP at epoch 10. The official baseline final result at epoch 24 is mAP 0.6044, so the current run still needs later-epoch validation before judging the final gain.

### Next-experiment decision

- No second experiment has been launched yet; the cleanaug VGGT run remains the priority reference.
- The current run is single-frame: `queue_length=1` and `use_student_history_bev=False`. It uses only the current-frame pre-LSS VGGT feature loss.
- The next priority is **history-only temporal VGGT-to-BEV distillation**, not a full clean/no-VGGT run: use the current frame as the student input, fuse the preceding 3 VGGT cache frames into the current ego coordinate frame as the teacher target, and apply an extra BEV distillation loss. This preserves causal inference (no future frames) and avoids the heavy memory/throughput cost of training the student with a multi-frame input queue.
- The repository already contains `TemporalVGGTBEVDistiller`; it traces `prev_idx` from the train annotation file, warps each cached frame to the current lidar/ego frame using `lidar2global`, and can therefore run with `queue_length=1`. A short 4-GPU smoke test is needed before full training to measure memory/step time and verify nonzero `loss_vggt_feat`.
- Do not run a second complete 4-GPU job concurrently with the current run; wait for its epoch-12 validation and use freed GPUs for the smoke test or subsequent full temporal run.

- [Official MapTRv2 6-GPU reproduction prep | 2026-08-12] 从零复现官方 `maptrv2_nusc_r50_24ep.py` 配方：
  - config: `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_official_6gpu_repro.py`
  - GPU: `1,2,3,4,5,6`；GPU 0、7 当时为其他用户任务占用
  - 官方每卡 batch `4` 保持不变；global batch 从官方 32 调整到 24，AdamW `lr` 按线性规则从 `6e-4` 调整到 `4.5e-4`
  - 除本机必要兼容设置外保持官方：本地 ResNet checkpoint 路径、只用 TextLogger，DDP timeout 3 小时以容纳 map evaluation

- [VGGT clean-view controlled run prep | 2026-08-24] 为消除 frozen clean VGGT cache 与 student augmentation 的视图不一致，新增 `maptrv2_nusc_r50_24ep_vggt_prelss_maptrsync_official4_cleanaug.py`：
  - 在官方 MapTRv2 设置上只加入 `pre-LSS` maptrsync VGGT 蒸馏，并关闭 `PhotoMetricDistortionMultiViewImage` 和 `GridMask`
  - 保留 FP16、官方几何/depth/task losses；可用 GPU 为 `0,1,5,6`，每卡 batch=4，global batch=16，LR 从官方 `6e-4` 线性缩放为 `3e-4`
  - 注意：应再配套运行同样无增强、无 VGGT 的 baseline，才能严格测量 VGGT 的净贡献

- [Clean augmentation ablation prep | 2026-08-10] 为验证 `pre-LSS VGGT distill` 是否被 student-only augment mismatch 拖累，新增一个最小对照配置：
  - config: `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss_maptrsync_fp32_cleanaug.py`
  - 相比当前 `maptrsync_fp32` 版，仅做两处改动：
    - 移除 `PhotoMetricDistortionMultiViewImage`
    - 关闭 `model.use_grid_mask`
  - 其余保持不变：仍使用对齐后的 `maptrsync` cache、同样的 `pre-LSS` 蒸馏位置、同样的损失权重和 24 epoch 训练设定
  - 目的：先确认 teacher/student 图像增强不一致是否是当前蒸馏效果受限的主因
  - launch: GPU `1,2,3,4,5,6`，6 卡；GPU 0、7 当时已被其他用户任务占用
  - background session: `tmux maptr_cleanaug_aug10`; work dir: `work_dirs/maptrv2_vggt_prelss_maptrsync_fp32_cleanaug_6gpu_aug10`

## 1. Storage and data-root decisions

Because the original NAS path was unstable / inaccessible, the working data root was moved to `/data1/zhangzj26`.

Active paths:

- nuScenes root: `/home/zhangzj26/MapTR/data/nuscenes -> /data1/zhangzj26/nuScenes_full`
- CAN bus root: `/home/zhangzj26/MapTR/data/can_bus -> /data1/zhangzj26/maptr_data/can_bus`
- downloaded checkpoint: `/home/zhangzj26/maptr_ckpts/maptr_tiny_r50_24e.pth`

Important note:

- repo `ckpts/` symlink still points to a broken NAS path, so evaluation used the explicit checkpoint path above instead of `./ckpts/...`.

## 2. Dataset blockers found and fixed

### 2.1 CAN bus missing / broken

Problem:

- `MapTR/data/can_bus` originally pointed to a hanging NAS path.

Fix:

- Downloaded official CAN bus archive from:
  - `https://motional-nuscenes.s3.amazonaws.com/public/v1.0/can_bus.zip`
- Extracted to:
  - `/data1/zhangzj26/maptr_data/can_bus`
- Repointed symlink:
  - `/home/zhangzj26/MapTR/data/can_bus -> /data1/zhangzj26/maptr_data/can_bus`

### 2.2 nuScenes map expansion missing

Problem:

- `maps/expansion` JSONs were missing under the nuScenes data root.

Fix:

- Downloaded official map expansion archive from:
  - `https://motional-nuscenes.s3.amazonaws.com/public/v1.0/nuScenes-map-expansion-v1.3.zip`
- Extracted into:
  - `/data1/zhangzj26/nuScenes_full/maps`

## 3. nuScenes preprocessing completed

Reference command from repo docs (`docs/prepare_dataset.md`):

```bash
python tools/maptrv2/custom_nusc_map_converter.py \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0 \
  --canbus ./data
```

Resulting generated files:

- `/data1/zhangzj26/nuScenes_full/nuscenes_map_infos_temporal_train.pkl` (399M, generated Jul 13 22:29)
- `/data1/zhangzj26/nuScenes_full/nuscenes_map_infos_temporal_val.pkl` (85M, generated Jul 13 22:30)
- `/data1/zhangzj26/nuScenes_full/nuscenes_map_infos_temporal_test.pkl` (85M, generated Jul 13 22:57)

Important compatibility note:

- MapTR v1 configs use `nuscenes_infos_temporal_{train,val}.pkl`.
- MapTRv2 preprocessing generated `nuscenes_map_infos_temporal_{train,val,test}.pkl`.
- During evaluation we overrode `data.test.ann_file` explicitly to point to the generated `nuscenes_map_infos_temporal_val.pkl`.

## 4. Pretrained checkpoint download

User requested trying HF mirror instead of Google Drive.

Working mirror page:

- `https://hf-mirror.com/jy137956/maptr/tree/main`

Checkpoint downloaded:

- file: `maptr_tiny_r50_24e.pth`
- direct URL used:
  - `https://hf-mirror.com/jy137956/maptr/resolve/main/maptr_tiny_r50_24e.pth?download=true`
- saved to:
  - `/home/zhangzj26/maptr_ckpts/maptr_tiny_r50_24e.pth`
- size verified:
  - `432796504 bytes` (`413M` by `ls -lh`)

Download command used:

```bash
wget -c --content-disposition \
  'https://hf-mirror.com/jy137956/maptr/resolve/main/maptr_tiny_r50_24e.pth?download=true' \
  -O /home/zhangzj26/maptr_ckpts/maptr_tiny_r50_24e.pth
```

## 5. Evaluation commands used

Repo reference (`docs/train_eval.md`):

```bash
./tools/dist_test_map.sh ./projects/configs/maptr/maptr_tiny_r50_24e.py ./path/to/ckpts.pth 8
```

Actual command used for this setup:

```bash
PORT=29625 bash tools/dist_test_map.sh \
  ./projects/configs/maptr/maptr_tiny_r50_24e.py \
  /home/zhangzj26/maptr_ckpts/maptr_tiny_r50_24e.pth \
  8 \
  --out /home/zhangzj26/MapTR/work_dirs/maptr_tiny_r50_24e_eval_8gpu.pkl \
  --cfg-options \
    data.workers_per_gpu=0 \
    data.test.ann_file=/data1/zhangzj26/nuScenes_full/nuscenes_map_infos_temporal_val.pkl \
    data.test.map_ann_file=/home/zhangzj26/MapTR/work_dirs/nuscenes_map_anns_val_8gpu.json
```

1-GPU distributed smoke-test command used before the 8-GPU launch:

```bash
PORT=29624 bash tools/dist_test_map.sh \
  ./projects/configs/maptr/maptr_tiny_r50_24e.py \
  /home/zhangzj26/maptr_ckpts/maptr_tiny_r50_24e.pth \
  1 \
  --out /home/zhangzj26/MapTR/work_dirs/maptr_tiny_r50_24e_eval_dist1.pkl \
  --cfg-options \
    data.workers_per_gpu=0 \
    data.test.ann_file=/data1/zhangzj26/nuScenes_full/nuscenes_map_infos_temporal_val.pkl \
    data.test.map_ann_file=/home/zhangzj26/MapTR/work_dirs/nuscenes_map_anns_val_dist1.json
```

Why `data.workers_per_gpu=0` was used:

- with the default multi-worker dataloader, the run hit:

```text
TypeError: cannot pickle 'dict_keys' object
```

Other evaluation note:

- `tools/test.py` contains `assert False` in the non-distributed branch, so even a 1-GPU test must still go through `dist_test_map.sh`.

## 6. Local compatibility fixes applied during reproduction

These changes were needed so the current local code and the downloaded checkpoint could run together in this environment.

### 6.1 Existing local modifications already present

The repo already had several local modifications before / during this reproduction, including:

- `tools/maptrv2/custom_nusc_map_converter.py`
  - patched to tolerate missing CAN bus and zero-fill fallback
- some local environment compatibility tweaks related to imports / numba / FORCE_CUDA

### 6.2 New fixes applied in this run

#### File: `projects/mmdet3d_plugin/maptr/modules/transformer.py`

Added compatibility handling so encoder outputs can be parsed whether they are:

- a plain tensor,
- a tuple/list, or
- a dict containing `bev` and optional `depth`.

Reason:

- the current local code path mixed MapTR v1 / MapTRv2-style return formats and previously crashed with:

```text
TypeError: new(): invalid data type 'str'
```

#### File: `projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py`

Added compatibility handling for transformer outputs of length 5:

- `(bev_embed, depth, hs, init_reference, inter_references)`

instead of the original length-4 expectation.

Reason:

- previous run crashed with:

```text
ValueError: too many values to unpack (expected 4)
```

## 7. Observed evaluation behavior

### 7.1 Checkpoint load

The checkpoint does load, but it prints a non-fatal mismatch warning:

```text
The model and loaded state dict do not match exactly
unexpected key in source state_dict: pts_bbox_head.transformer.encoder.layers.0.attentions.1.attention.grid_offsets
```

This warning did not prevent the evaluation loop from starting.

### 7.2 1-GPU smoke test

After the compatibility fixes, the distributed 1-GPU test successfully entered the validation loop and progressed to:

- `10/6019`

This confirmed that:

- dataset access works,
- checkpoint loading works well enough to run inference,

## 8. Temporal VGGT-BEV distillation scaffold (A2)

Goal for the next baseline:

- start from `MapTRv2`
- use a `history 3-frame` VGGT teacher: `{t-2, t-1, t}`
- keep the student on the `current frame only`
- keep inference unchanged

Files added / updated for the first A2 implementation:

- `projects/mmdet3d_plugin/maptr/distill/temporal_vggt_bev_distill.py`
- `projects/mmdet3d_plugin/maptr/distill/__init__.py`
- `projects/mmdet3d_plugin/maptr/__init__.py`
- `projects/mmdet3d_plugin/maptr/detectors/maptrv2.py`
- `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_a2.py`

Current implementation behavior:

- loads per-sample VGGT cache by `sample_idx`
- expects cache entries containing:
  - `feature` or `vggt_feat`
  - `point_map`
  - optional `confidence`
- warps each history frame from its own lidar frame into the current lidar frame using `lidar2global`
- pools lifted VGGT features into BEV
- fuses the 3-frame teacher BEV by weighted averaging
- distills the current-frame MapTRv2 `bev_embed`

Important bug fixed during this pass:

- the initial scaffold had an error in temporal BEV fusion:
  - per-frame BEV features were already accumulated as weighted sums
  - later fusion multiplied them by the weights again
- this was corrected so the temporal teacher now performs a proper weighted average instead of double-counting dense cells

## 9. REPA-inspired loss design used in A2

Reference checked:

- `third_party/REPA/loss.py`

Main idea borrowed:

- project teacher/student features
- normalize them
- use cosine alignment as the main representation loss

Practical adaptation in this repo:

- replaced plain `1x1 conv` alignment with a small projector:
  - `1x1 conv -> GELU -> 1x1 conv`
- kept a cosine term as the primary loss
- kept a light normalized L1 term as a stabilizer
- masked distillation with:
  - valid teacher BEV cells
  - `gt_seg_mask` map region

## 10. A2 config for the first training run

Config:

- `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_a2.py`

Current choices:

- `queue_length=3`
- `use_student_history_bev=False`
- `samples_per_gpu=2`
- VGGT cache root placeholder:
  - `/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_p37`

Note:

- this config is the minimal history-teacher version only
- it does **not** yet include:
  - future-frame teacher
  - temporal attention
  - Gaussian splat
  - uncertainty weighting

## 11. Validation status

Local validation completed:

- `python -m py_compile` passed for:
  - `projects/mmdet3d_plugin/maptr/distill/temporal_vggt_bev_distill.py`
  - `projects/mmdet3d_plugin/maptr/detectors/maptrv2.py`
  - `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_a2.py`
- a synthetic forward smoke-test also passed for:
  - `TemporalVGGTBEVDistiller`
  - random cache tensors + random BEV student feature
- a real 1-sample GPU smoke-test passed for the new offline exporter:
  - output file size: about `2.2M`
  - cached tensor shapes:
    - `feature`: `(6, 128, 37, 37)`
    - `point_map`: `(6, 37, 37, 3)`
    - `confidence`: `(6, 37, 37)`

Still missing before full training:

- run a first smoke training step with the A2 config
- check memory use because queue images are still collated even when the student history path is disabled
- the main remaining issue is not CAN bus / map expansion / missing PKL generation.

## 12. Offline VGGT cache exporter

New script:

- `tools/maptrv2/export_vggt_cache.py`

Design choices:

- use local `VGGT` checkpoint:
  - `/data1/zhangzj26/vggt_ckpts/model.pt`
- avoid `track_head` dependency issues by loading:
  - `aggregator`
  - `depth_head`
  - remapped `track_head.feature_extractor`
- apply compatibility patches for this local environment:
  - old torch `autocast`
  - old torch `meshgrid`
  - disable fused attention path that requires newer `scaled_dot_product_attention`

Stored cache format per sample token:

- `feature`: `float16`, `(6, 128, 37, 37)`
- `point_map`: `float32`, `(6, 37, 37, 3)`
- `confidence`: `float16`, `(6, 37, 37)`

Why `37x37`:

- much smaller than dense `259x259` maps
- keeps storage manageable
- still preserves per-camera spatial structure aligned with the ViT patch grid

Estimated storage:

- observed smoke-test file size: about `2.2M / sample`
- for the full train split this is on the order of `~60GB`, which is acceptable under `/data1`

## 13. Multi-GPU cache export launch

User approved placing the cache under `/data1/zhangzj26`.

Created directories:

- `/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_p37`
- `/data1/zhangzj26/maptr_logs`

Launched background export in tmux across 3 shards:

- session `vggt_cache_train_g0`
  - GPU `0`
  - range `[0, 9377)`
- session `vggt_cache_train_g1`
  - GPU `1`
  - range `[9377, 18754)`
- session `vggt_cache_train_g7`
  - GPU `7`
  - range `[18754, 28130)`

Log files:

- `/data1/zhangzj26/maptr_logs/vggt_cache_train_g0.log`
- `/data1/zhangzj26/maptr_logs/vggt_cache_train_g1.log`
- `/data1/zhangzj26/maptr_logs/vggt_cache_train_g7.log`

## 14. Distillation teacher fixes after code review

A follow-up review identified three high-impact issues in the first A2 scaffold, and they were addressed without invalidating the already-exported per-token VGGT cache:

- `projects/mmdet3d_plugin/maptr/distill/temporal_vggt_bev_distill.py`
  - teacher frame selection no longer trusts the randomized dataset queue
  - now traces the fixed `prev` chain from `ann_file` to recover true continuous history frames
  - for A2 this now matches `{t-2, t-1, t}` much more faithfully
- `projects/mmdet3d_plugin/maptr/distill/temporal_vggt_bev_distill.py`
  - added simple near-ground filtering in `point_to_bev`
  - current default:
    - `ground_height_range=(-3.0, 1.0)`
  - this is meant to reduce dynamic / non-map contamination before teacher fusion
- `projects/mmdet3d_plugin/maptr/distill/temporal_vggt_bev_distill.py`
  - missing teacher cache for one sample no longer drops the whole batch
  - distillation is now computed only on valid samples inside the batch
- `projects/mmdet3d_plugin/maptr/distill/temporal_vggt_bev_distill.py`
  - teacher reliability mask now uses per-cell multi-frame support instead of a pure binary valid mask

Config updates:

- `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_a2.py`
  - added:
    - `ann_file=/home/zhangzj26/MapTR/data/nuscenes/nuscenes_map_infos_temporal_train.pkl`
    - `ground_height_range=(-3.0, 1.0)`

Important note:

- the current VGGT cache export does **not** need to be redone for these fixes
- cache files are still keyed by `sample_idx` / token and remain reusable

### 7.3 8-GPU evaluation status

The 8-GPU run was launched in tmux:

- tmux session: `maptr_eval8`
- log file: `/home/zhangzj26/MapTR/work_dirs/maptr_tiny_r50_24e_eval_8gpu.log`

Observed progress before failure:

- reached `32/6019`

Failure recorded at `2026-07-14 00:42:54`:

```text
RuntimeError: CUDA out of memory. Tried to allocate 72.00 MiB (GPU 5; 23.68 GiB total capacity; 388.19 MiB already allocated; 23.00 MiB free; 410.00 MiB reserved in total by PyTorch)
RuntimeError: CUDA out of memory. Tried to allocate 72.00 MiB (GPU 2; 23.68 GiB total capacity; 388.19 MiB already allocated; 5.00 MiB free; 410.00 MiB reserved in total by PyTorch)
```

Interpretation:

- the 8-GPU failure is now a GPU-memory availability problem on at least GPU 2 and GPU 5,
- not a dataset-format / CAN bus / map expansion issue.

## 8. Current reproducible status summary

Completed:

- CAN bus downloaded and linked correctly
- map expansion downloaded and placed correctly
- nuScenes temporal map info PKLs generated successfully
- pretrained MapTR tiny checkpoint downloaded from HF mirror
- evaluation path validated via 1-GPU distributed smoke test
- current local code patched to handle transformer output compatibility

Current blocker for full 8-GPU eval:

- CUDA OOM on GPUs 2 and 5 during the distributed validation run

## 9. Suggested next reproduction steps

1. Re-run evaluation when all 8 GPUs are actually free.
2. If multi-user GPU occupancy remains unstable, use a smaller GPU set explicitly, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 PORT=29630 bash tools/dist_test_map.sh \
  ./projects/configs/maptr/maptr_tiny_r50_24e.py \
  /home/zhangzj26/maptr_ckpts/maptr_tiny_r50_24e.pth \
  4 \
  --out /home/zhangzj26/MapTR/work_dirs/maptr_tiny_r50_24e_eval_4gpu.pkl \
  --cfg-options \
    data.workers_per_gpu=0 \
    data.test.ann_file=/data1/zhangzj26/nuScenes_full/nuscenes_map_infos_temporal_val.pkl \
    data.test.map_ann_file=/home/zhangzj26/MapTR/work_dirs/nuscenes_map_anns_val_4gpu.json
```

3. After eval is stable, move on to training with the repo reference command:

```bash
bash tools/dist_train.sh ./projects/configs/maptr/maptr_tiny_r50_24e.py 8
```

4. For training in this environment, likely keep data under `/data1/zhangzj26/...` and record any config overrides in this same log file for later comparison.

## 10. Official baseline target for current eval

From `README.md`, the closest baseline for the current checkpoint/config pair is:

- config: `projects/configs/maptr/maptr_tiny_r50_24e.py`
- checkpoint: `maptr_tiny_r50_24e.pth`
- expected metric: `mAP 50.0`
- reported speed: `15.1 FPS`
- reported memory: `10287M (bs 4)`

This is the main reference point to compare against once validation finishes successfully.

## 11. Latest blocker snapshot (2026-07-14)

The latest 8-GPU validation attempt did not finish because the machine was already heavily occupied by other GPU jobs.

Observed GPU memory snapshot during investigation:

- GPU 0: `15173 / 24576 MiB`
- GPU 1: `3704 / 24576 MiB`
- GPU 2: `22265 / 24576 MiB`
- GPU 3: `11634 / 24576 MiB`
- GPU 4: `21933 / 24576 MiB`
- GPU 5: `22247 / 24576 MiB`
- GPU 6: `21627 / 24576 MiB`
- GPU 7: `1729 / 24576 MiB`

Conclusion:

- current failure is dominated by external GPU contention,
- the cleanest immediate workaround is to run eval only on relatively free GPUs, or wait until the 8 cards are actually free.

## 12. Background eval launched on GPUs 1 and 7

Per follow-up decision, a slower but more stable background eval was launched only on GPUs 1 and 7.

Command:

```bash
CUDA_VISIBLE_DEVICES=1,7 PORT=29631 bash tools/dist_test_map.sh \
  ./projects/configs/maptr/maptr_tiny_r50_24e.py \
  /home/zhangzj26/maptr_ckpts/maptr_tiny_r50_24e.pth \
  2 \
  --out /home/zhangzj26/MapTR/work_dirs/maptr_tiny_r50_24e_eval_gpu1_7.pkl \
  --cfg-options \
    data.workers_per_gpu=0 \
    data.test.ann_file=/data1/zhangzj26/nuScenes_full/nuscenes_map_infos_temporal_val.pkl \
    data.test.map_ann_file=/home/zhangzj26/MapTR/work_dirs/nuscenes_map_anns_val_gpu1_7.json
```

Background session / logs:

- tmux session: `maptr_eval_g17`
- log file: `/home/zhangzj26/MapTR/work_dirs/maptr_tiny_r50_24e_eval_gpu1_7.log`
- output pkl target: `/home/zhangzj26/MapTR/work_dirs/maptr_tiny_r50_24e_eval_gpu1_7.pkl`

### 12.1 First 2-GPU run outcome

The first `GPU 1,7` run completed all `6019` samples, but the script still failed at the end because `tools/test.py` had another hard-coded `assert False` after result collection.

Observed behavior:

- full inference reached `6019/6019`
- result writing message appeared
- process then crashed at `tools/test.py:240`

Fix applied:

- `tools/test.py`
  - replaced the hard-coded `assert False` with `mmcv.dump(outputs, args.out)`

### 12.2 Restarted 2-GPU eval after test-script fix

Restarted session:

- tmux session: `maptr_eval_g17_v2`
- log file: `/home/zhangzj26/MapTR/work_dirs/maptr_tiny_r50_24e_eval_gpu1_7_v2.log`

Early observed speed after restart:

- around `5.3 ~ 5.4 it/s`
- ETA around `1040 ~ 1050s` at the beginning of the rerun

## 13. MapTRv2 baseline checkpoint recovery attempt

Goal:

- switch the main baseline from MapTR v1 to `MapTRv2` on nuScenes and evaluate the official `24ep` R50 model.

### 13.1 Official MapTRv2 checkpoint link recovered from git history

The current `README.md` on this branch no longer exposes the actual v2 download URL, but the old repository history still contains it.

Recovered from commit:

- commit: `853702a75831fe8cbd1f8abc04303d964f49493a`
- message: `upload maptrv2_nusc_r50_24e checkpoint&log`

Recovered official links:

- checkpoint: `https://drive.google.com/file/d/1AmQ3fT-J-MM4B8kh_9Gm2G5guM92Agww/view?usp=sharing`
- log: `https://drive.google.com/file/d/1rrAXza6FTYUs8kfr5126qWU6-FNGGMwD/view?usp=sharing`

Expected file naming:

- target checkpoint path: `/home/zhangzj26/maptr_ckpts/maptrv2_nusc_r50_24e.pth`
- config: `projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py`

### 13.2 Download-source probing results

Google Drive connectivity from this server is currently blocked / timing out.

Observed direct probes:

- `drive.google.com`: connection timeout
- `docs.google.com`: connection timeout
- `drive.usercontent.google.com`: connection timeout

So the official v2 checkpoint link is known, but the server cannot download it directly at the moment.

### 13.3 Alternative mirror search results

Searches performed:

- `hf-mirror.com/api/models?search=maptr`
- `hf-mirror.com/api/models?search=maptrv2`
- `hf-mirror.com/api/datasets?search=maptr`
- `hf-mirror.com/api/datasets?search=maptrv2_nusc`

Findings:

- HF mirror contains:
  - `jy137956/maptr` with `maptr_tiny_r50_24e.pth` only
  - `LIvanoff/maptrv2-av2-predictions` (predictions, not nuScenes checkpoint)
  - `ericaw/mapbevprediction_maptrv2_nusc_r50_24ep_w_centerline` (processed prediction tensors, not the official checkpoint)
- no public HF-mirror repository with the official `maptrv2_nusc_r50_24e.pth` checkpoint was found during this round.

### 13.4 Host-side GPU visibility check

Inside the sandbox, CUDA devices are hidden, so GPU checks must run with escalated host access.

Observed host GPU snapshot during this recovery round:

- GPU 0: `3185 / 24576 MiB`
- GPU 1: `1979 / 24576 MiB`
- GPU 2: `20540 / 24576 MiB`
- GPU 3: `9909 / 24576 MiB`
- GPU 4: `20208 / 24576 MiB`
- GPU 5: `20522 / 24576 MiB`
- GPU 6: `20156 / 24576 MiB`
- GPU 7: `7841 / 24576 MiB`

Conclusion for the next step:

- once the `MapTRv2` checkpoint is available locally, `GPU 1` is still the cleanest immediate choice,
- but evaluation is currently blocked by checkpoint download availability rather than by code or dataset readiness.

## 14. Third-party geometry / representation libraries prepared

User-provided snapshots were placed under:

- `/home/zhangzj26/MapTR/third_party/vggt`
- `/home/zhangzj26/MapTR/third_party/REPA`

Quick structural verification:

- both repos contain `README.md` and `requirements.txt`
- `VGGT` also contains `pyproject.toml`
- `VGGT` package tree is present under `vggt/`
- `REPA` training / model scripts are present under repo root and `models/`
- representative Python files passed `py_compile`

Notes:

- these are uploaded snapshots, not full git clones (`.git` metadata absent)
- for development / integration usage they are structurally usable
- if exact upstream commit tracking is needed later, we should record commit hashes separately

## 15. VGGT pretrained checkpoint background download

According to the local `VGGT` README, the default pretrained checkpoint is:

- official model id: `facebook/VGGT-1B`
- official file: `model.pt`

Because direct access to `huggingface.co` is unstable from this server, a reachable mirror endpoint was used for downloading the same model artifact:

- mirror URL used: `https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt`

Background download setup:

- tmux session: `vggt_ckpt_dl`
- target dir: `/data1/zhangzj26/vggt_ckpts`
- target file: `/data1/zhangzj26/vggt_ckpts/model.pt`
- log file: `/data1/zhangzj26/vggt_ckpts/download.log`

Progress can be checked with:

```bash
tmux attach -t vggt_ckpt_dl
```

or

```bash
tail -f /data1/zhangzj26/vggt_ckpts/download.log
```

## 16. VGGT cache export finished for A2 distillation

Final cache root:

- `/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_p37`

Completion status:

- exported cache files: `28130 / 28130`
- cache format is per-sample-token `.pt`, so later teacher-frame logic changes can reuse the same cache

Retry recovery:

- original failed token list: `/home/zhangzj26/MapTR/work_dirs/vggt_cache_retry_failed_tokens_g7.txt`
- missing-only retry list: `/home/zhangzj26/MapTR/work_dirs/vggt_cache_retry_missing_tokens_g7.txt`
- retry log: `/data1/zhangzj26/maptr_logs/vggt_cache_retry_g7.log`

## 17. A2 distillation implementation and training compatibility fixes

### 17.1 New A2 config

Main config used for this round:

- `/home/zhangzj26/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_a2.py`

Key local settings in that config:

- uses cache root `/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_p37`
- overrides broken repo pretrained path with:
  - `/home/zhangzj26/maptr_ckpts/resnet50-19c8e357.pth`
- disables `TensorboardLoggerHook` and keeps `TextLoggerHook` only

### 17.2 Distillation logic refinements

Primary implementation file:

- `/home/zhangzj26/MapTR/projects/mmdet3d_plugin/maptr/distill/temporal_vggt_bev_distill.py`

Important fixes applied:

- teacher frame selection changed to explicit fixed-history tracing through sample `prev` links
- teacher BEV pooling adds near-ground filtering with `ground_height_range`
- missing cache now skips per sample instead of dropping the whole batch
- teacher validity now keeps support weighting instead of pure binary valid mask
- distill path now handles `gt_seg_mask` arriving as a per-image tensor list during training

Related integration points:

- detector wiring: `/home/zhangzj26/MapTR/projects/mmdet3d_plugin/maptr/detectors/maptrv2.py`
- plugin export: `/home/zhangzj26/MapTR/projects/mmdet3d_plugin/maptr/__init__.py`
- train-entry compatibility patch: `/home/zhangzj26/MapTR/tools/train.py`

## 18. Smoke training and full training launch

### 18.1 Smoke training

Purpose:

- verify that `MapTRv2 + A2 history-3 VGGT distillation` can really enter the training loop on this host

Smoke ann subset:

- `/home/zhangzj26/MapTR/work_dirs/smoke_nuscenes_map_infos_temporal_train_32.pkl`

Successful smoke work dir:

- `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_a2_smoke4`

Observed successful training signal:

- smoke run completed `1 epoch / 32 iters`
- distillation loss appeared in logs as `loss_vggt_feat`
- example logged values during the run:
  - iter `5/32`: `loss_vggt_feat: 0.0428`
  - iter `30/32`: `loss_vggt_feat: 0.0090`

Blockers found before the successful smoke run:

1. `mmcv` / `yapf` incompatibility in `FormatCode(..., verify=True)`
2. broken repo `ckpts/` path for `resnet50-19c8e357.pth`
3. `TensorboardLoggerHook` importing `torch.utils.tensorboard` and hitting:
   - `AttributeError: module 'distutils' has no attribute 'version'`
4. distill path assuming `gt_seg_mask` was already a batched tensor

### 18.2 First full-training launch failure

Initial 2-GPU background launch used GPUs `1,7`, but it failed immediately because:

- `data.workers_per_gpu=2` triggered:
  - `TypeError: cannot pickle 'dict_keys' object`

Conclusion:

- for this environment, full training must also force `data.workers_per_gpu=0`

### 18.3 Current active full-training job

That early full-training launch is no longer the recommended one.

## 19. Full-training failure analysis and final stable baseline launch

### 19.1 2-GPU OOM root cause

When switching back to the full dataset with the original full-train batch size:

- GPUs: `1,7`
- `data.samples_per_gpu=2`
- `data.workers_per_gpu=2`
- no validation during train

the run failed with:

```text
RuntimeError: CUDA out of memory. Tried to allocate 150.00 MiB
```

Conclusion:

- on this host, `MapTRv2 + A2 VGGT distill` full training is not stable at `2 GPUs x 2 samples_per_gpu`

### 19.2 2-GPU reduced-batch NCCL failure

Next attempt:

- GPUs: `1,7`
- `data.samples_per_gpu=1`
- `data.workers_per_gpu=2`
- `--no-validate`

This did enter training and printed:

- `Epoch [1][10/14065]`

but later failed with:

```text
Watchdog caught collective operation timeout
```

and finally:

```text
terminate called after throwing an instance of 'std::runtime_error'
```

Conclusion:

- 2-GPU reduced-batch training is not reliable enough on the full dataset in the current environment

### 19.3 Stable full-data baseline now running

The currently recommended / active stable training job is:

- tmux session: `maptrv2_a2_train_1gpu`
- GPU: `7`
- work dir: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_a2_train_1gpu_b2_noval`
- log file: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_a2_train_1gpu_b2_noval.log`

Launch settings:

- `samples_per_gpu=2`
- `workers_per_gpu=2`
- `--no-validate`
- `log_config.interval=10`
- `checkpoint_config.interval=1`

Launch command pattern:

```bash
CUDA_VISIBLE_DEVICES=7 PORT=29652 \
bash /home/zhangzj26/MapTR/tools/dist_train.sh \
  /home/zhangzj26/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_a2.py 1 \
  --no-validate \
  --work-dir /home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_a2_train_1gpu_b2_noval \
  --cfg-options \
    data.samples_per_gpu=2 \
    data.workers_per_gpu=2 \
    log_config.interval=10 \
    checkpoint_config.interval=1
```

Latest confirmed runtime status:

- training is alive in tmux
- first logs were printed successfully:
  - `Epoch [1][10/14065]`
  - `Epoch [1][20/14065]`
  - `Epoch [1][30/14065]`
- representative speed after warmup:
  - around `1.85s` to `2.46s` per iter on the stable 1-GPU run

Practical implication:

- this is slower than the ideal multi-GPU plan,
- but it is the first full-data baseline run that is verified to keep training stably in this environment

## 20. Temporal VGGT distill visualization export

Goal:

- directly inspect whether the new teacher-side design is actually doing something useful:
  - near-ground filtering via `ground_height_range`
  - support-weighted `teacher_valid`
  - `use_map_mask` gating

New script:

- added `tools/analysis/visualize_temporal_vggt_distill.py`

What it exports per token:

- `before_valid`: teacher BEV valid cells before z filtering
- `removed_by_ground`: cells removed by `ground_height_range`
- `removed_outside_map`: removed cells that also lie outside GT map mask
- `after_valid`: teacher valid cells after z filtering
- `teacher_support`: per-cell support ratio across history frames
- `teacher_valid_weighted`: support-weighted valid mask
- `map_mask`: GT map region used by `use_map_mask`
- `distill_mask`: final mask that actually supervises the student

Run command used:

```bash
PATH=/home/zhangzj26/miniconda3/envs/maptr/bin:$PATH \
python /home/zhangzj26/MapTR/tools/analysis/visualize_temporal_vggt_distill.py \
  --tokens \
    d59d3c7fb1a1445785274b1a08b5daa1 \
    cc1567e902724f82898f405601923c3c \
    0e0afd22bec441c084b8a52393a558eb \
  --output-dir /home/zhangzj26/MapTR/work_dirs/distill_debug_cases_train3 \
  --device cpu
```

Outputs:

- figure dir: `/home/zhangzj26/MapTR/work_dirs/distill_debug_cases_train3`
- summary json: `/home/zhangzj26/MapTR/work_dirs/distill_debug_cases_train3/summary.json`

Quick observations from these exported cases:

- `d59d3c7fb1a1445785274b1a08b5daa1`
  - z filter changed nothing
  - but support weighting still matters:
    - `support_1frame_cells=29`
    - `support_2frame_cells=28`
    - `support_3frame_cells=440`
  - final distill mask shrank from `497` teacher-valid cells to `204` map-supervised cells

- `0e0afd22bec441c084b8a52393a558eb`
  - again almost no z-filter effect
  - support weighting still separated weak vs strong cells:
    - `18 / 25 / 447` for `1f / 2f / 3f`
  - final distill mask shrank from `490` teacher-valid cells to `235` map-supervised cells

- `cc1567e902724f82898f405601923c3c`
  - z filter had a visible effect:
    - `before_valid_cells=624`
    - `after_valid_cells=494`
    - `removed_by_ground_cells=130` (`20.8%`)
    - `removed_outside_map_cells=86` (`66.2%` of removed cells)
  - this supports the claim that ground filtering can remove high-z / non-map clutter
  - but it also showed that z filtering alone does not guarantee better map alignment:
    - `before_on_map_ratio=10.9%`
    - `after_on_map_ratio=4.9%`

Current conclusion from the visualization stage:

- `use_map_mask` is clearly effective at concentrating supervision onto actual map regions
- support weighting is also clearly doing something meaningful because many cells are only supported by 1 or 2 frames
- near-ground filtering is useful as a noise suppression step, but its benefit is scene-dependent
- based on current exported cases, z filtering alone should be treated as a helper, not a full fix for teacher/map misalignment


## 21. Switched distillation from BEV teacher to pre-LSS image-feature teacher

Change in direction:

- stop distilling a hand-built VGGT BEV teacher into student BEV
- keep original MapTRv2 `LSSTransform` unchanged
- distill frozen VGGT multi-view features into student `img_neck` features before LSS

Code changes:

- detector now supports `pre_lss_distill_cfg`:
  - `/home/zhangzj26/MapTR/projects/mmdet3d_plugin/maptr/detectors/maptrv2.py`
- new distiller:
  - `/home/zhangzj26/MapTR/projects/mmdet3d_plugin/maptr/distill/vggt_feature_distill.py`
- distill package export updated:
  - `/home/zhangzj26/MapTR/projects/mmdet3d_plugin/maptr/distill/__init__.py`
- old temporal BEV distiller now skips cache read `OSError` instead of crashing immediately

Current pre-LSS alignment strategy:

- teacher: VGGT cache `feature` tensor per camera (`6 x 128 x 37 x 37`)
- student: current-frame `img_neck` feature before LSS (`B x 6 x 256 x H x W`)
- alignment used for now:
  - match camera order by camera name if filenames are available in `img_meta`
  - resize teacher feature/confidence to the student spatial size with bilinear interpolation
  - use normalized cosine + L1 distillation, confidence-weighted

New config prepared:

- `/home/zhangzj26/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss.py`

4-GPU training attempt:

- target GPUs: `0,1,4,7`
- tmux session: `maptrv2_vggt_prelss_g0147`
- work dir: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_train_g0147`
- log file: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_train_g0147.log`

Observed failure:

- this time the crash was no longer from VGGT cache loading inside the distillation branch
- the startup failure came from base nuScenes data I/O on the storage itself:
  - `OSError: [Errno 5] Input/output error: 'data/nuscenes/nuscenes_map_infos_temporal_train.pkl'`
  - image reads also failed with `Input/output error` on files under `data/nuscenes/samples/...`

Interpretation:

- the new pre-LSS distillation code path is in place
- but the full 4-GPU run is currently blocked by unstable `/data1` nuScenes storage access, not by the new distillation logic

## 22. July 16, 2026 - resumed 4-GPU pre-LSS distillation training after /data1 recovery

Storage recovery check:

- `/data1` recovered and is readable again on July 16, 2026.
- `df -h /data1` succeeded and reported about `6.3T` free space.
- `data/nuscenes/nuscenes_map_infos_temporal_train.pkl` is readable again.

Config update before relaunch:

- removed the temporary `local_data/*.pkl` annotation override from
  `/home/zhangzj26/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss.py`
- the pre-LSS config now falls back to the base config's original
  `data/nuscenes/nuscenes_map_infos_temporal_{train,val}.pkl` paths

Launch details:

- target GPUs: `0,1,4,7`
- tmux session: `maptrv2_vggt_prelss_g0147_jul16`
- config: `/home/zhangzj26/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss.py`
- work dir: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_train_g0147_jul16`
- log file: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_train_g0147_jul16.log`
- launch command uses `CUDA_VISIBLE_DEVICES=0,1,4,7` and keeps distillation before LSS

## 23. July 16, 2026 - launched epoch-1 pre-LSS evaluation on GPU 6

Goal:

- measure the early validation accuracy of the new pre-LSS VGGT distillation run
- use the saved checkpoint from epoch 1 while training continues on GPUs `0,1,4,7`

Launch details:

- eval GPU: `6`
- tmux session: `maptrv2_vggt_prelss_eval_g6_e1`
- config: `/home/zhangzj26/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss.py`
- checkpoint: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_train_g0147_jul16/epoch_1.pth`
- output pkl: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_epoch1_eval_gpu6.pkl`
- log file: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_epoch1_eval_gpu6.log`
- launch command: `CUDA_VISIBLE_DEVICES=6 PORT=29636 bash tools/dist_test_map.sh ... 1 --out ... --cfg-options data.workers_per_gpu=2`

Notes:

- `tools/dist_test_map.sh` forces distributed test mode and appends `--eval chamfer`
- evaluation is launched from `epoch_1.pth` because training was started with `--no-validate`

## 24. July 16, 2026 - epoch-1 pre-LSS evaluation finished on GPU 6

Checkpoint and run:

- config: `/home/zhangzj26/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss.py`
- checkpoint: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_train_g0147_jul16/epoch_1.pth`
- eval log: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_epoch1_eval_gpu6.log`
- result pkl: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_epoch1_eval_gpu6.pkl`

Final reported metrics:

- `NuscMap_chamfer/mAP = 0.11989762442600398`
- `NuscMap_chamfer/divider_AP = 0.12753340105215707`
- `NuscMap_chamfer/ped_crossing_AP = 0.050415104099859796`
- `NuscMap_chamfer/boundary_AP = 0.181744368125995`

Threshold-specific APs from the final dict:

- `divider`: `thr_0.5=0.034377455711364746`, `thr_1.0=0.12569403648376465`, `thr_1.5=0.22252871096134186`
- `ped_crossing`: `thr_0.5=0.0011303848586976528`, `thr_1.0=0.029778670519590378`, `thr_1.5=0.12033625692129135`
- `boundary`: `thr_0.5=0.03420688584446907`, `thr_1.0=0.18818669021129608`, `thr_1.5=0.32283952832221985`

Important interpretation note:

- the evaluation log prints an AP table for the current threshold block; the last visible table showed `mAP = 0.222` under the `thr_1.5` block
- the actual overall metric returned by `dataset.evaluate(...)` is the final dict value:
  `NuscMap_chamfer/mAP = 0.11989762442600398`
- for experiment comparison, use the final dict values rather than the single-threshold table

Quick takeaway:

- this is an `epoch_1` early-check result only, not the final converged number
- relative weakness is currently strongest on `ped_crossing`
- `boundary` is the strongest of the three map classes at this stage

## 25. July 16, 2026 - launched epoch-2 pre-LSS evaluation

Goal:

- evaluate the `epoch_2.pth` checkpoint while the full training continues in background
- compare early `epoch_2` quality against the already measured `epoch_1` result

Launch details:

- config: `/home/zhangzj26/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss.py`
- checkpoint: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_train_g0147_jul16/epoch_2.pth`
- tmux session: `maptrv2_vggt_prelss_eval_g6_e2`
- output pkl: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_epoch2_eval_gpu6.pkl`
- log file: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_epoch2_eval_gpu6.log`
- launch command uses `tools/dist_test_map.sh` and still evaluates `chamfer`

## 26. Epoch-8 evaluation watcher armed

Current state when requested:

- training has entered `epoch 8`, but `epoch_8.pth` is not saved yet
- latest saved checkpoint is still `epoch_7.pth`

Watcher details:

- tmux session: `maptrv2_vggt_prelss_eval_g6_e8_wait`
- behavior: poll until `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_train_g0147_jul16/epoch_8.pth` appears, then auto-launch eval
- target eval GPU: `6`
- output pkl: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_epoch8_eval_gpu6.pkl`
- log file: `/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_epoch8_eval_gpu6.log`

- [Log check] 日志最新时间戳显示为 2026-07-17 07:43:54；训练仍在 Epoch 8（约 1430/7033），`epoch_8.pth` 尚未生成。
- [Eval watcher] 已重新挂起 tmux session `maptrv2_vggt_prelss_eval_g6_e8_wait`，等待 `epoch_8.pth` 出现后自动在 GPU 6 上启动评测，输出到 `work_dirs/maptrv2_vggt_prelss_epoch8_eval_gpu6.log`。

- [Epoch 7 eval launch] 已在 tmux session `maptrv2_vggt_prelss_epoch7_eval_g235` 后台启动 3 卡评测：`CUDA_VISIBLE_DEVICES=2,3,5`，checkpoint=`epoch_7.pth`，日志输出到 `work_dirs/maptrv2_vggt_prelss_epoch7_eval_gpu235.log`，结果输出到 `work_dirs/maptrv2_vggt_prelss_epoch7_eval_gpu235.pkl`。

- [Epoch 7 eval fix] 发现首次 3 卡评测命令直接调用 `./tools/dist_test.sh` 时因脚本无执行权限失败；已改为 `bash tools/dist_test.sh`，并于日志时间戳约 07:56 重新启动 `maptrv2_vggt_prelss_epoch7_eval_g235`。同时把 `epoch 8` watcher 也同步修正为 `bash tools/dist_test.sh`。

- [Epoch 7 eval fix 2] 发现后台 shell 未进入 `maptr` 环境，导致 `torch.distributed.launch` 找不到 `torch`；已改为 `PATH=/home/zhangzj26/miniconda3/envs/maptr/bin:$PATH` 后重新启动 `epoch_7` 评测，并同步修正 `epoch_8` watcher。注意：当前部分日志文件内部时间戳写成 `2026-07-17`，相对当前日期 `2026-07-16` 提前一天，说明服务器或日志时间存在偏差。

- [Epoch 7/8 eval fix 3 | 2026-07-17] 确认报错根因是误用了 `tools/dist_test.sh`，该脚本固定追加 `--eval bbox`；而 `nuscenes_offlinemap_dataset.py` 只支持 `chamfer/iou`。已切换为 `bash tools/dist_test_map.sh` 重启 `epoch_7` 三卡评测，并同步修复 `epoch_8` watcher。

- [Epoch 7 eval result | 2026-07-17] `epoch_7.pth` 三卡评测已完成，结果文件 `work_dirs/maptrv2_vggt_prelss_epoch7_eval_gpu235.pkl` 已生成。最终指标：`NuscMap_chamfer/mAP=0.02908942169730248`，`divider_AP=0.032273056296010814`，`ped_crossing_AP=0.005953489775241867`，`boundary_AP=0.04904171902065476`。相较 `epoch 1`（0.1199）明显更差，但高于 `epoch 2`（0.00233）。

- [Restart | 2026-07-17] 用户决定优先检查 fp16 是否导致 `grad_norm=nan`。已停止旧 tmux 训练/评测会话：`maptrv2_vggt_prelss_g0147_jul16`、`maptrv2_vggt_prelss_eval_g6_e8_wait`、`maptrv2_vggt_prelss_epoch7_eval_g235`；已删除旧 checkpoint：`work_dirs/maptrv2_vggt_prelss_train_g0147_jul16/epoch_7.pth` 与 `latest.pth`。新增配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss_fp32.py`（`fp16=None`），并在 tmux session `maptrv2_vggt_prelss_fp32_g0147_jul17` 中使用 GPU `0,1,4,7` 重新启动训练，work dir 为 `work_dirs/maptrv2_vggt_prelss_fp32_train_g0147_jul17`。

- [FP32 epoch 2 eval launch | 2026-07-17] 已在 tmux session `maptrv2_vggt_prelss_fp32_e2_eval_g235` 后台启动 `epoch_2.pth` 三卡地图评测：`CUDA_VISIBLE_DEVICES=2,3,5`，配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss_fp32.py`，日志输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch2_eval_gpu235.log`，结果输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch2_eval_gpu235.pkl`。

- [FP32 epoch 2 eval result | 2026-07-17] `epoch_2.pth` 三卡地图评测已完成，结果文件 `work_dirs/maptrv2_vggt_prelss_fp32_epoch2_eval_gpu235.pkl` 已生成。最终指标：`NuscMap_chamfer/mAP=0.034693634751987626`，`divider_AP=0.03665053968628248`，`ped_crossing_AP=0.00115042550430348`，`boundary_AP=0.06627993906537692`。相比旧 fp16 版 `epoch 2`（0.0023256）有明显回升，但仍低于旧 fp16 版 `epoch 1`（0.1198976）与 fp16 版 `epoch 7`（0.0290894）略高。

- [FP32 epoch 4 eval launch | 2026-07-17] 已在 tmux session `maptrv2_vggt_prelss_fp32_e4_eval_g235` 后台启动 `epoch_4.pth` 三卡地图评测：`CUDA_VISIBLE_DEVICES=2,3,5`，配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss_fp32.py`，日志输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch4_eval_gpu235.log`，结果输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch4_eval_gpu235.pkl`。

- [FP32 epoch 4 eval result | 2026-07-17] `epoch_4.pth` 三卡地图评测已完成，结果文件 `work_dirs/maptrv2_vggt_prelss_fp32_epoch4_eval_gpu235.pkl` 已生成。最终指标：`NuscMap_chamfer/mAP=0.13914869444367164`，`divider_AP=0.13418364276488623`，`ped_crossing_AP=0.05275658170770233`，`boundary_AP=0.23050585885842642`。相较 FP32 `epoch 2`（0.0346936）大幅提升，也略高于旧 fp16 `epoch 1`（0.1198976）。

- [FP32 epoch 6 eval launch | 2026-07-17] 已在 tmux session `maptrv2_vggt_prelss_fp32_e6_eval_g235` 后台启动 `epoch_6.pth` 三卡地图评测：`CUDA_VISIBLE_DEVICES=2,3,5`，配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss_fp32.py`，日志输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch6_eval_gpu235.log`，结果输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch6_eval_gpu235.pkl`。

- [FP32 epoch 6 eval result | 2026-07-18] `epoch_6.pth` 三卡地图评测已完成，结果文件 `work_dirs/maptrv2_vggt_prelss_fp32_epoch6_eval_gpu235.pkl` 已生成。最终指标：`NuscMap_chamfer/mAP=0.17457929496756855`，`divider_AP=0.17499295497934023`，`ped_crossing_AP=0.08323674509301782`，`boundary_AP=0.2655081848303477`。相较 FP32 `epoch 4`（0.1391487）继续提升。

- [FP32 epoch 8 eval launch | 2026-07-17] 已在 tmux session `maptrv2_vggt_prelss_fp32_e8_eval_g235` 后台启动 `epoch_8.pth` 三卡地图评测：`CUDA_VISIBLE_DEVICES=2,3,5`，配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss_fp32.py`，日志输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch8_eval_gpu235.log`，结果输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch8_eval_gpu235.pkl`。

- [FP32 epoch 8 eval result | 2026-07-18] `epoch_8.pth` 三卡地图评测已完成，结果文件 `work_dirs/maptrv2_vggt_prelss_fp32_epoch8_eval_gpu235.pkl` 已生成。最终指标：`NuscMap_chamfer/mAP=0.1599545837768043`，`divider_AP=0.1477046236395836`，`ped_crossing_AP=0.07238617628657569`，`boundary_AP=0.25977295140425366`。相较 FP32 `epoch 6`（0.1745793）略有回落。

- [FP32 epoch 10 eval launch | 2026-07-18] 已在 tmux session `maptrv2_vggt_prelss_fp32_e10_eval_g235` 后台启动 `epoch_10.pth` 三卡地图评测：`CUDA_VISIBLE_DEVICES=2,3,5`，配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss_fp32.py`，日志输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch10_eval_gpu235.log`，结果输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch10_eval_gpu235.pkl`。

- [FP32 epoch 10 eval result | 2026-07-18] `epoch_10.pth` 三卡地图评测已完成，结果文件 `work_dirs/maptrv2_vggt_prelss_fp32_epoch10_eval_gpu235.pkl` 已生成。最终指标：`NuscMap_chamfer/mAP=0.17859838514899215`，`divider_AP=0.1653714825709661`，`ped_crossing_AP=0.0982198747806251`，`boundary_AP=0.27220379809538525`。相较 FP32 `epoch 8`（0.1599546）回升，也略高于 FP32 `epoch 6`（0.1745793）。

- [FP32 epoch 12 eval launch | 2026-07-18] 已在 tmux session `maptrv2_vggt_prelss_fp32_e12_eval_g235` 后台启动 `epoch_12.pth` 三卡地图评测：`CUDA_VISIBLE_DEVICES=2,3,5`，配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss_fp32.py`，日志输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch12_eval_gpu235.log`，结果输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch12_eval_gpu235.pkl`。

- [FP32 epoch 12 eval result | 2026-07-18] `epoch_12.pth` 三卡地图评测已完成，结果文件 `work_dirs/maptrv2_vggt_prelss_fp32_epoch12_eval_gpu235.pkl` 已生成。最终指标：`NuscMap_chamfer/mAP=0.23030462861061096`，`divider_AP=0.2234410122036934`，`ped_crossing_AP=0.14937450736761093`，`boundary_AP=0.31809836626052856`。相较 FP32 `epoch 10`（0.1785984）有明显提升。

- [FP32 epoch 14 eval launch | 2026-07-18] 已在 tmux session `maptrv2_vggt_prelss_fp32_e14_eval_g235` 后台启动 `epoch_14.pth` 三卡地图评测：`CUDA_VISIBLE_DEVICES=2,3,5`，配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss_fp32.py`，日志输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch14_eval_gpu235.log`，结果输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch14_eval_gpu235.pkl`。

- [FP32 epoch 14 eval result | 2026-07-18] `epoch_14.pth` 三卡地图评测已完成，结果文件 `work_dirs/maptrv2_vggt_prelss_fp32_epoch14_eval_gpu235.pkl` 已生成。最终指标：`NuscMap_chamfer/mAP=0.24085170403122902`，`divider_AP=0.21963170419136682`，`ped_crossing_AP=0.18015494818488756`，`boundary_AP=0.3227684597174327`。相较 FP32 `epoch 12`（0.2303046）继续提升。

- [FP32 epoch 20 eval launch | 2026-07-18] 已在 tmux session `maptrv2_vggt_prelss_fp32_e20_eval_g235` 后台启动 `epoch_20.pth` 三卡地图评测：`CUDA_VISIBLE_DEVICES=2,3,5`，配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss_fp32.py`，日志输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch20_eval_gpu235.log`，结果输出到 `work_dirs/maptrv2_vggt_prelss_fp32_epoch20_eval_gpu235.pkl`。

- [FP32 epoch 20 eval result | 2026-07-19] `epoch_20.pth` 三卡地图评测已完成，结果文件 `work_dirs/maptrv2_vggt_prelss_fp32_epoch20_eval_gpu235.pkl` 已生成。最终指标：`NuscMap_chamfer/mAP=0.27546339854598045`，`divider_AP=0.2599417343735695`，`ped_crossing_AP=0.21277044340968132`，`boundary_AP=0.35367801785469055`。相较 FP32 `epoch 14`（0.2408517）继续提升。

- [Official MapTRv2 baseline launch | 2026-07-19] 新增配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_g25_autoeval.py`，保持官方 `MapTRv2` 配方，仅把 `checkpoint_config.max_keep_ckpts` 调整为 `12` 以保留更多自动评测点；已在 tmux session `maptrv2_official_g25_train_jul19` 中用 GPU `2,5` 后台启动训练，命令基于官方 `maptrv2_nusc_r50_24ep.py` 并加 `--autoscale-lr`，work dir 为 `work_dirs/maptrv2_official_r50_24ep_g25_jul19`，外层日志为 `work_dirs/maptrv2_official_r50_24ep_g25_jul19.log`。该配置本身每 `2` epoch 自动保存并自动地图评测一次。

- [Official MapTRv2 baseline relaunch | 2026-07-19] 发现该仓库 `tools/train.py` 会在分布式 world size 初始化前执行 `--autoscale-lr`，导致 2 卡训练被错误缩放到 `7.5e-05`；已停止刚才的错误启动并修正 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_g25_autoeval.py`，显式设置 `optimizer.lr=1.5e-4`（相对官方 8 卡线性缩放到 2 卡），随后在同一 tmux session `maptrv2_official_g25_train_jul19` 中重新启动训练。

- [Official MapTRv2 baseline eval prep | 2026-07-19] 由于两卡官方 baseline 训练在进入首个 iteration 前提前退出，现改为先复现官方 checkpoint 的直接评测。将使用 README / gitextract 中给出的 Google Drive 官方权重 `maptrv2_nusc_r50_24ep.pth` 下载到 `/home/zhangzj26/maptr_ckpts/maptrv2_nusc_r50_24ep_official.pth` 后进行地图评测。

- [Official MapTRv2 4-GPU baseline launch | 2026-07-19] 已按用户要求清理失败的官方 baseline 调试目录：`work_dirs/maptrv2_official_r50_24ep_g25_jul19*` 与 `work_dirs/maptrv2_official_single_gpu_debug_jul19`；新增配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_g2457_autoeval.py`，保持官方 MapTRv2 配方，仅调整为 `samples_per_gpu=2`、`workers_per_gpu=0`、`optimizer.lr=1.5e-4`，并保留每 `2` epoch 自动保存/自动评测。已在 tmux session `maptrv2_official_g2457_train_jul19` 中用 GPU `2,4,5,7` 后台启动训练，work dir 为 `work_dirs/maptrv2_official_r50_24ep_g2457_jul19`，外层日志为 `work_dirs/maptrv2_official_r50_24ep_g2457_jul19.log`。

- [Official MapTRv2 4-GPU relaunch fix | 2026-07-19] 定位到官方 baseline 卡住的根因是 `MapTR/ckpts -> /nas1/.../ckpts`，而 `/nas1` 当前超时，导致日志一直停在 `load checkpoint from local path: ckpts/resnet50-19c8e357.pth`。已将派生配置切换为显式使用本地权重 `/home/zhangzj26/maptr_ckpts/resnet50-19c8e357.pth`，清理失败的四卡目录后，在 tmux session `maptrv2_official_g2457_train_jul19_fix` 中重新启动 2/4/5/7 四卡官方 baseline 训练。

- [Local ckpt path switch | 2026-07-19] 为彻底绕开 `/nas1` 异常，已将 ResNet50 预训练权重复制到本地可写目录 `MapTR/data/ckpts/resnet50-19c8e357.pth`，后续官方 baseline / 派生配置统一改走该本地路径，不再依赖 `MapTR/ckpts -> /nas1/...`。

- [Official MapTRv2 4-GPU data-ckpt relaunch | 2026-07-19] 已将官方 baseline 的 backbone 预训练权重切到本地 `MapTR/data/ckpts/resnet50-19c8e357.pth` 后，重新在 tmux session `maptrv2_official_g2457_train_jul19_datafix` 中用 GPU `2,4,5,7` 启动训练；work dir 仍为 `work_dirs/maptrv2_official_r50_24ep_g2457_jul19`，保持每 2 epoch 自动保存和自动地图评测。

- [Official MapTRv2 4-GPU relaunch fix 2 | 2026-07-19] 继续定位到第二个阻塞点：训练已越过模型/数据初始化，但在 `before_run` 阶段被 `TensorboardLoggerHook` 卡死，报错为 `AttributeError: module distutils has no attribute version`。已在 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_g2457_autoeval.py` 中临时移除 TensorBoard logger，仅保留 `TextLoggerHook`，随后重新启动四卡 baseline。

- [Official MapTRv2 baseline status | 2026-07-20 15:30 CST] 四卡官方 baseline 当前可见最新训练进度为 `Epoch [6][700/3517]`，日志文件为 `work_dirs/maptrv2_official_r50_24ep_g2457_jul19.log`。截至目前已完成自动评测的 checkpoint 有：
  - `epoch 2`: `NuscMap_chamfer/mAP=0.23980959587626985`，`divider_AP=0.23314542571703592`，`ped_crossing_AP=0.17581108212471008`，`boundary_AP=0.3104722797870636`
  - `epoch 4`: `NuscMap_chamfer/mAP=0.3407379413644473`，`divider_AP=0.31670961777369183`，`ped_crossing_AP=0.30231282860040665`，`boundary_AP=0.40319137771924335`
  - 当前 best checkpoint 为 `work_dirs/maptrv2_official_r50_24ep_g2457_jul19/best_NuscMap_chamfer/mAP_epoch_4.pth`

- [Official vs VGGT pre-LSS quick compare | 2026-07-20] 和我们已经跑完的 `VGGT pre-LSS distill` 分支相比，官方 baseline 早期明显更强：
  - 官方 `epoch 2 mAP=0.2398` vs VGGT `epoch 2 mAP=0.0347`
  - 官方 `epoch 4 mAP=0.3407` vs VGGT `epoch 4 mAP=0.1391`
  - 甚至官方 `epoch 4 mAP=0.3407` 仍高于 VGGT `epoch 20 mAP=0.2755`
  - 这说明当前 `VGGT pre-LSS distill` 方案虽然能学起来，但距离超过 MapTRv2 baseline 还有明显差距，后续重点应放在蒸馏位置、teacher 表达密度、以及与原生多视角/LSS 特征对齐方式上。

- [Official MapTRv2 baseline stop status | 2026-07-21] 复查四卡官方 baseline 发现训练并未完整跑完 24 epoch：`epoch_6.pth` 已在 `2026-07-20 18:07` 成功保存，但随后训练/评测流程在 `2026-07-20 18:52` 异常退出。当前日志里没有看到 `epoch 6` 的最终地图评测指标，进程也已经结束。异常形式为 `tools/train.py FAILED`，`rank 1/2/3` 收到 `SIGABRT (-6)`。因此目前 baseline 只能算“部分复现成功”，已稳定复现到 `epoch 4` 指标、并跑到 `epoch 6` checkpoint，但整条 24-epoch baseline 还没有完全收尾。

- [Official MapTRv2 baseline resume analysis | 2026-07-21] 进一步定位发现，`epoch 6` 之后的退出根因并不是训练本身崩掉，而是 `CustomDistEvalHook` 触发地图评测后，rank 0 长时间停在 CPU 端结果格式化 / 评测，其他 rank 很快返回并继续进入训练，最终在下一次 DDP `all_reduce` 处等待超过默认 `NCCL` 30 分钟超时。日志关键信息：
  - `Saving checkpoint at 6 epochs` 发生在 `2026-07-20 18:07:20`
  - 随后 `val/.../nuscmap_results.json` 开始格式化
  - `2026-07-20 18:52` 出现 `Watchdog caught collective operation timeout` 与 `RuntimeError: NCCL communicator was aborted`

- [Official MapTRv2 baseline resume launch | 2026-07-21 15:53 CST] 已开始尝试从 `epoch_6.pth` 续训，目标是把 24 epoch baseline 跑完：
  - 续训 checkpoint: `work_dirs/maptrv2_official_r50_24ep_g2457_jul19/epoch_6.pth`
  - 续训 work dir: `work_dirs/maptrv2_official_r50_24ep_g2457_jul19`
  - GPU: `2,4,5,7`
  - 外层日志: `work_dirs/maptrv2_official_r50_24ep_g2457_jul21_resume_fix2.log`
  - 新的内部训练日志: `work_dirs/maptrv2_official_r50_24ep_g2457_jul19/20260721_155302.log`
  - 当前已确认日志出现：`load checkpoint from local path: .../epoch_6.pth` 与 `resumed epoch 6, iter 21102`

- [Resume compatibility fixes | 2026-07-21] 为了让续训真正起得来，额外做了两处兼容修复：
  - `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_g2457_autoeval.py` 中把 `dist_params.timeout` 扩到 `3` 小时，避免地图评测期间再次触发默认 30 分钟 `NCCL` 超时
  - `tools/train.py` 中加入 config dump / config logging 的 fallback，绕过 `timedelta` 无法被 `mmcv` 的 `pretty_text` 正常序列化的问题

- [VGGT pre-LSS minimal alignment fix | 2026-07-21] 针对“VGGT cache 与 MapTRv2 训练图像几何预处理不一致”的问题，已先实现最小修复版：
  - `tools/maptrv2/export_vggt_cache.py` 新增 `preprocess_mode=maptr_train_geom`，默认改为使用 MapTR 几何链条导出 cache：原图 -> 固定 `0.5` resize -> pad 到 `32` 的倍数
  - 为满足 VGGT patch-size `14` 的要求，只在导出时额外做临时右/下 padding，并在 VGGT feature/depth 输出后再裁回 MapTR 有效区域，避免旧版 `square 518x518` 预处理带来的几何错位
  - cache 导出的空间尺寸不再强制为正方形，而是按 MapTR 处理后图像的纵横比生成，例如 `480x800 -> 22x37`
  - 新增配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss_maptrsync_fp32.py`，预留新的 cache 根目录：`/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_maptrsync_p37`

- [VGGT cache alignment spot-check | 2026-07-21] 已对 `nuscenes_map_infos_temporal_train.pkl` 中 3 个样本（index `0/1000/20000`）做几何预处理对齐抽检，并进一步修正 exporter 使其与 MapTR 完全一致：
  - 修正点：`tools/maptrv2/export_vggt_cache.py` 的 `load_and_preprocess_images_maptr` 原先使用 `PIL bilinear resize`，像素值与 MapTR pipeline 的 `mmcv.imresize` 有系统差异；现已改为 `mmcv.imresize + mmcv.impad_to_multiple`
  - 抽检相机：`CAM_FRONT` / `CAM_FRONT_LEFT` / `CAM_BACK`
  - 抽检结果：MapTR 有效区域内像素级完全一致，`max diff = 0`、`mean diff = 0`、`non-zero ratio = 0`
  - 当前几何链条可视为已经对齐：原图 `900x1600` -> 缩放后 `450x800` -> MapTR 有效输入 `480x800` -> 仅为适配 VGGT patch-size 临时补到 `490x812`，而保存前会裁回 `480x800`
  - 结论：现在这版 `maptrsync` cache 至少在图像几何预处理层面已经和 MapTR train pipeline 对齐，后续如果效果仍差，优先怀疑蒸馏目标/位置本身，而不是这一步的 resize-pad 错位

- [Old VGGT cache bug confirmed | 2026-07-21] 已确认上一版 `VGGT pre-LSS` cache 存在实现级对齐错误，需要和新版本区分看待：
  - 旧版 exporter 走的是 `square 518x518` 预处理，且 resize 使用 `PIL`；MapTR 训练真实输入则是 `mmcv.imresize(scale=0.5) + pad_to_32`
  - 因此旧版 teacher 与 student 在进入 `pre-LSS distill` 前看到的图像几何并不一致，不是简单的尺寸不同，而是有效像素位置分布本身不同
  - 抽检样本中，把旧版结果仅为可视化而 resize 回 `480x800` 后，与 MapTR 有效输入相比仍有明显偏差：`mean diff` 约 `2~5`，`non-zero ratio` 约 `0.68~0.76`
  - 这类错位不会直接把训练跑崩，但非常容易出现“loss 能下降、mAP 缓慢上涨、却长期显著落后 baseline”的现象；因此此前 `VGGT pre-LSS` 分支结果不能直接用于判断方法本身无效
  - 对应可视化已保存到 `work_dirs/vggt_alignment_check_20260721/old_vs_new_alignment_sample_*.png`

- [Cache refresh decision | 2026-07-21] 对后续 `maptrsync pre-LSS distill` 实验，需要重新导出 cache，不能继续复用旧版 square-cache：
  - 必须重刷的原因：旧 cache 的 feature / depth / confidence 都建立在错误图像预处理之上，哪怕训练代码换成新版本，对齐问题也不会自动消失
  - 建议保留旧 cache 仅作为历史对照，不再作为新实验输入
  - 新实验应统一使用新根目录 `/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_maptrsync_p37`
  - 最稳妥的流程是先小规模补刷并跑短程 sanity check，再决定是否全量重刷 + 全量训练

- [Full maptrsync cache refresh launch/fix | 2026-07-21] 已按“避开 baseline 训练的 `2/4/5/7` 卡”思路，先用 `0/1/3/6` 四卡并行启动全量重刷，分片区间分别为 `0:7033`、`7033:14066`、`14066:21098`、`21098:28130`：
  - tmux session: `vggt_cache_maptrsync_g0/g1/g3/g6`
  - 实时日志目录：`work_dirs/vggt_cache_maptrsync_logs/`
  - 输出目录：`/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_maptrsync_p37`

- [Exporter compatibility fix | 2026-07-21] 全量重刷第一次尝试很快发现环境兼容问题：
  - 失败现象：日志中连续出现 `interpolate() got an unexpected keyword argument 'antialias'`
  - 根因：当前环境的 `torch` 版本较老，`VGGT` 第三方代码里的 `F.interpolate(..., antialias=...)` 不兼容
  - 修复方式：在 `tools/maptrv2/export_vggt_cache.py` 中增加了 `F.interpolate` 兼容包装，并同时修正了旧版 `torch.meshgrid` 对 `indexing='xy'` 的兼容实现，避免位置编码网格维度错乱
  - 首次失败日志已保留到 `work_dirs/vggt_cache_maptrsync_logs/attempt1_fail_antialias/`

- [24GB OOM workaround for maptrsync export | 2026-07-21] 第二次尝试发现 `maptrsync` 预处理后的输入分辨率更大（如 `490x812`），在 24GB `RTX 3090` 上按 6 视角一次性前向 `VGGT` 会 OOM：
  - 失败现象：`CUDA out of memory. Tried to allocate 8.89 GiB`
  - 判断：这是 `maptrsync` 分辨率变大后的新问题，不是旧 square-cache 那条链路里的问题
  - 处理：给 exporter 增加 `--max-views-per-forward`，并在实际脚本里设为 `1`，改成“每个 camera view 顺序前向、最后再拼回 6 视角 cache”
  - 这样做是一个工程性折中：它不再让 VGGT 在 6 个摄像头之间做联合 attention，但对于当前 `pre-LSS per-view distill` 目标，优先保证缓存可稳定导出
  - 第二次失败日志已保留到 `work_dirs/vggt_cache_maptrsync_logs/attempt2_fail_oom/`

- [View-chunk smoke test pass | 2026-07-21] 在正式第三次重启全量任务前，已用单样本 smoke test 验证新 exporter 可以成功导出：
  - smoke 输出目录：`/tmp/vggt_maptrsync_smoke`
  - 结果：成功生成 `e93e98b63d3b40209056d129dc53ceee.pt`，大小约 `1.3M`
  - 说明 `antialias` 兼容、`meshgrid` 兼容、以及 `1-view forward` 的 OOM 规避都已打通

- [Full maptrsync cache refresh relaunch | 2026-07-21 18:07 CST] 已基于修复后的 exporter 第三次重启四卡全量重刷：
  - session 仍为 `vggt_cache_maptrsync_g0/g1/g3/g6`
  - 当前脚本参数包含 `--max-views-per-forward 1`
  - 各 shard 目前都已打印启动行：`[VGGT cache] device=cuda:0, samples=...`

- [MapTR-sync cache refresh complete | 2026-07-24] 新版 `pre-LSS` 对齐 cache 已全量补齐完成：
  - cache root: `/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_maptrsync_p37`
  - final count: `28130 / 28130`
  - 8 卡补刷最终结果无失败；缺失样本主要来自 `g1/g3/g5/g7`，其余 shard 基本为直接 skip
  - 可继续使用配置 `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_pre_lss_maptrsync_fp32.py` 做正式训练

- [Pre-LSS feature alignment status | 2026-07-24] 关于“上个版本空间尺度没对齐，是否需要再导出对应 feature 检查”：
  - 旧版 square-cache 的确存在几何错位，不能再拿来判断方法效果
  - 新版 `maptrsync` cache 已在图像几何层面对齐到 MapTR train pipeline，抽检可视化见 `work_dirs/vggt_alignment_check_20260721/alignment_sample_*.png`
  - 当前 `pre-LSS` 蒸馏实现里还会在 forward 时把 teacher feature resize 到 student feature 的空间尺寸后再算 loss，见 `projects/mmdet3d_plugin/maptr/distill/vggt_feature_distill.py`
  - 因此现在如果精度仍不理想，优先怀疑蒸馏目标本身是否有用，而不是继续怀疑这一版 cache 的 resize/pad 空间尺度不一致

- [A2 temporal 3-frame DDP launch fix | 2026-08-25] 已在 GPU `0,1` 用 tmux session `maptr_a2_temporal3_g01` 启动 `history-only 3-frame VGGT BEV teacher` 训练：
  - config: `projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_prelss_maptrsync_official4_cleanaug_a2_temporal3.py`
  - work dir: `work_dirs/maptrv2_a2_temporal3_g01_aug25`
  - 训练日志: `work_dirs/maptrv2_a2_temporal3_g01_aug25/launch_20260825_2101_stable_ddp.log`
  - 启动初期的阻塞并非数据盘/cache 问题。首轮 `forward_train`、pre-LSS distill、MapTR head、temporal distill 都能在约 `0.6~1.4s` 内完成，但 DDP 会在后续 iteration 失败/阻塞。
  - 根因 1：LSS encoder 不会读取 BEVFormer-only 的 `row_embed`、`col_embed`、`level_embeds`、`cams_embeds`、`can_bus_mlp`，但这些参数默认仍为可训练状态。A2 显式冻结该组参数，并使用 `find_unused_parameters=False`，避免 DDP 的 unused-parameter 图遍历阻塞。
  - 根因 2：当某个 batch/rank 没有有效 temporal teacher（缺 cache 或地图 mask 无有效像素）时，`temporal_distiller.student_proj` 原先不会进入计算图；已改为返回与 `student_proj` 相连的零值 loss。pre-LSS distiller 也加入同样保护。
  - 同时冻结两个 teacher projector，并放进 `torch.no_grad()`，因为 teacher 输出本来就会 `detach`，不应被 DDP 当成待训练分支。
  - 验证：训练已越过此前必现的第二轮错误，并打印 `Epoch [1][10/7033]`、`[20/7033]`、`[30/7033]`。当前约 `5.8~6.1s/iter`，batch size 为每卡 `2`，日志 ETA 约 `11~12` 天。
