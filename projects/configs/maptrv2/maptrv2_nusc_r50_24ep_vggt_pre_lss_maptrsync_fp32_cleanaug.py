_base_ = ['./maptrv2_nusc_r50_24ep_vggt_pre_lss_maptrsync_fp32.py']

# Clean augmentation ablation for pre-LSS VGGT distillation:
# - disable student-only photometric distortion
# - disable grid mask
# Keep all other training settings identical to the aligned maptrsync run.

model = dict(
    use_grid_mask=False,
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='CustomPointToMultiViewDepth',
        downsample=1,
        grid_config=dict(
            x=[-30.0, -30.0, 0.15],
            y=[-15.0, -15.0, 0.15],
            z=[-10, 10, 20],
            depth=[1.0, 35.0, 0.5])),
    dict(type='PadMultiViewImageDepth', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=['divider', 'ped_crossing', 'boundary']),
    dict(type='CustomCollect3D', keys=['img', 'gt_depth'])
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
    ),
)
