from datetime import timedelta

_base_ = ['./maptrv2_nusc_r50_24ep.py']

# Official MapTRv2 recipe reproduced on six 24 GB GPUs.
# The original recipe is 8 GPUs x 4 samples with lr=6e-4. Preserve the
# per-GPU batch and linearly scale the learning rate for global batch 24.
model = dict(
    pretrained=dict(img='/home/zhangzj26/MapTR/data/ckpts/resnet50-19c8e357.pth'),
)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
)
optimizer = dict(lr=4.5e-4)
checkpoint_config = dict(interval=2, max_keep_ckpts=12)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
# Rank 0 evaluates the map metrics after every checkpoint; prevent slower
# CPU-side evaluation from tripping the default NCCL timeout.
dist_params = dict(backend='nccl', timeout=timedelta(hours=3))

