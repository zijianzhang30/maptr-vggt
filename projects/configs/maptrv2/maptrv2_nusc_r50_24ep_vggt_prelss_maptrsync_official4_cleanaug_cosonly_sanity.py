_base_ = ['./maptrv2_nusc_r50_24ep_vggt_prelss_maptrsync_official4_cleanaug_cosonly.py']

# 20-iteration preflight only; model/data/augmentation remain inherited.
runner = dict(type='IterBasedRunner', max_iters=20)
evaluation = dict(interval=1000000)
checkpoint_config = dict(interval=1000000, max_keep_ckpts=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
work_dir = '/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_maptrsync_official4_cleanaug_cosonly_sanity20'
