_base_ = ['./maptrv2_nusc_r50_24ep.py']

# Keep the official MapTRv2 training recipe, but retain more checkpoints so
# the auto-eval results every 2 epochs remain easy to compare later.
model = dict(pretrained=dict(img='/home/zhangzj26/MapTR/data/ckpts/resnet50-19c8e357.pth'))
checkpoint_config = dict(interval=2, max_keep_ckpts=12)

# Reduce per-GPU batch further for a 2-GPU baseline run and bake in the
# matching linear-scaled learning rate relative to the official 8-GPU recipe.
data = dict(samples_per_gpu=1, workers_per_gpu=2)
optimizer = dict(lr=3.75e-5)
