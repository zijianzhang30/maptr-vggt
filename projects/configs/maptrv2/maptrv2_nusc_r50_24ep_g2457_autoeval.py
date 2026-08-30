from datetime import timedelta

_base_ = ['./maptrv2_nusc_r50_24ep.py']

# Keep the official MapTRv2 recipe, but use a conservative 4-GPU setup that is
# more likely to stay stable on this server while still auto-saving/evaluating
# every 2 epochs.
model = dict(pretrained=dict(img='/home/zhangzj26/MapTR/data/ckpts/resnet50-19c8e357.pth'))
data = dict(samples_per_gpu=2, workers_per_gpu=0)
optimizer = dict(lr=1.5e-4)
checkpoint_config = dict(interval=2, max_keep_ckpts=12)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
# Rank 0 spends a long time in map evaluation, so extend the DDP timeout.
dist_params = dict(backend='nccl', timeout=timedelta(hours=3))
del timedelta
