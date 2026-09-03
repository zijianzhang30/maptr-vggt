_base_ = ['./maptrv2_nusc_r50_24ep_vggt_prelss_maptrsync_official4_cleanaug_cosonly_g03.py']

# REPA-style early distillation stop: keep the same single-frame cosine-only
# objective for epochs 1-16, then continue the MapTR task losses alone.
custom_hooks = [
    dict(
        type='DistillWeightScheduleHook',
        target='pre_lss_distiller',
        schedule=[
            (1, 0.05),
            (17, 0.0),
        ],
    ),
]

work_dir = '/data/public/zhangzj26/maptr_vggt_cosonly_cleanaug_stop16_g67'
