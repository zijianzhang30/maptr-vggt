# MapTR-VGGT environment setup on ucas-ai-14

This document reproduces the environment currently used on `ucas-ai-19` for
the MapTR-VGGT experiments. Commands assume the repository is located at
`/home/zhangzj26/MapTR`.

## 1. Requirements

- Linux x86_64 with NVIDIA GPU and a driver compatible with CUDA 11.1
- Miniconda/Anaconda
- GCC/G++ suitable for compiling PyTorch CUDA extensions
- Read access to the nuScenes data and the precomputed VGGT cache

The known-good software versions are:

| Component | Version |
|---|---|
| Python | 3.8.20 |
| PyTorch | 1.9.1+cu111 |
| torchvision | 0.10.1+cu111 |
| CUDA used to build PyTorch | 11.1 |
| MMCV | 1.4.0 |
| MMDetection | 2.14.0 |
| MMSegmentation | 0.14.1 |
| MMDetection3D | 0.17.2 |

Check the GPU driver first:

```bash
nvidia-smi
```

## 2. Create the conda environment

```bash
conda create -n maptr python=3.8 -y
conda activate maptr

pip install torch==1.9.1+cu111 \
  torchvision==0.10.1+cu111 \
  torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1
pip install timm shapely==1.8.5.post1
```

If a wheel is unavailable, do not silently install a newer OpenMMLab stack;
the old MapTR code depends on the versions above.

## 3. Install the repository and CUDA extensions

```bash
cd /home/zhangzj26/MapTR/mmdetection3d
python setup.py develop

cd /home/zhangzj26/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install

cd /home/zhangzj26/MapTR
pip install -r requirement.txt
```

If extension compilation fails, verify `gcc --version`, `nvcc --version`,
`CUDA_HOME`, and that the active Python is from the `maptr` environment.

## 4. Configure data paths

The configs use repository-relative `data/nuscenes` and `data/can_bus` paths.
On the source machine these resolve to:

```text
data/nuscenes -> /data1/zhangzj26/nuScenes_full
data/can_bus  -> /data1/zhangzj26/maptr_data/can_bus
```

The single-frame VGGT teacher features are read from:

```text
/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_maptrsync_p37
```

If ucas-ai-14 mounts the datasets elsewhere, either recreate the links or
change only the corresponding data/cache paths in the experiment config:

```bash
cd /home/zhangzj26/MapTR
mkdir -p data
ln -s /actual/path/to/nuScenes_full data/nuscenes
ln -s /actual/path/to/can_bus data/can_bus
```

The nuScenes directory must include the MapTR annotation files
`nuscenes_map_infos_temporal_train.pkl`,
`nuscenes_map_infos_temporal_val.pkl`, and `nuscenes_map_anns_val.json`.

## 5. Verify the installation

```bash
conda activate maptr
cd /home/zhangzj26/MapTR
python - <<'PY'
import torch, mmcv, mmdet, mmseg, mmdet3d
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('mmcv:', mmcv.__version__)
print('mmdet:', mmdet.__version__)
print('mmseg:', mmseg.__version__)
print('mmdet3d:', mmdet3d.__version__)
PY
```

Expected core versions are those in the table above. Also test imports of the
custom CUDA operators before starting a long run:

```bash
python - <<'PY'
from mmdet3d.ops import bev_pool
from projects.mmdet3d_plugin.maptr.modules.ops.geometric_kernel_attn import GeometricKernelAttention
print('custom operator imports: OK')
PY
```

## 6. Run the cosine-only experiment

The current controlled-comparison config is:

```text
projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_prelss_maptrsync_official4_cleanaug_cosonly.py
```

It uses single-frame VGGT pre-LSS distillation with
`cosine_weight=1.0`, `l1_weight=0.0`, and `loss_weight=0.05`; temporal BEV
distillation is disabled.

Example six-GPU launch:

```bash
cd /home/zhangzj26/MapTR
tmux new-session -s vggt_cosonly
bash work_dirs/launch_vggt_cosonly_cleanaug_0to5.sh
```

The supplied launcher selects GPUs 0-5. Detach from tmux with `Ctrl-b`, then
`d`; reconnect with `tmux attach -t vggt_cosonly`.

## 7. Common issues

- `Operation not permitted` during rendezvous: local TCP/multiprocess access
  is restricted by the execution environment; run from a normal shell.
- `ModuleNotFoundError` or undefined symbols: reactivate `maptr`, then rebuild
  `mmdetection3d` and `geometric_kernel_attn` in that environment.
- Missing VGGT loss or zero cache hit rate: verify `cache_root` and its read
  permissions.
- CUDA OOM: first confirm no unrelated GPU processes are running. Do not alter
  batch size or augmentation for a controlled comparison without recording a
  new experiment configuration.
