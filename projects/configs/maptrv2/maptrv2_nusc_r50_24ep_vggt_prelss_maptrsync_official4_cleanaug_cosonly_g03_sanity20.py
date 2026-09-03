_base_ = ['./maptrv2_nusc_r50_24ep_vggt_prelss_maptrsync_official4_cleanaug_cosonly_g03.py']

# Preflight only. Inherit the exact model, teacher, cache, and cleanaug data
# pipeline from the formal experiment.
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=20)
evaluation = dict(interval=1000000)
checkpoint_config = dict(interval=1000000, max_keep_ckpts=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
work_dir = '/data/public/zhangzj26/maptr_vggt_cosonly_cleanaug_g03_sanity20'
