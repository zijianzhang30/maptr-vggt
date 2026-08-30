_base_ = [
    './maptrv2_nusc_r50_24ep_vggt_prelss_maptrsync_official4_cleanaug.py'
]

# Test whether VGGT is most useful as early optimization guidance. All other
# settings inherit the completed single-frame clean-augmentation experiment.
custom_hooks = [
    dict(
        type='DistillWeightScheduleHook',
        target='pre_lss_distiller',
        schedule=[
            (1, 0.05),
            (7, 0.02),
            (13, 0.005),
            (19, 0.0),
        ],
    ),
]

work_dir = 'work_dirs/maptrv2_vggt_prelss_cleanaug_decay4_aug27'
