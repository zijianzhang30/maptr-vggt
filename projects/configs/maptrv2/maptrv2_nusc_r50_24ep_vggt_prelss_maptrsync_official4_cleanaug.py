from datetime import timedelta

_base_ = ['./maptrv2_nusc_r50_24ep.py']

# Controlled VGGT experiment: preserve the official MapTRv2 recipe except for
# the pre-LSS VGGT loss and the two student-only augmentations that cannot be
# reproduced by the frozen, clean teacher cache.
# Four GPUs x four samples => global batch 16; scale 6e-4 from the official
# global batch 32 linearly to 3e-4.
model = dict(
    pretrained=dict(img='/home/zhangzj26/MapTR/data/ckpts/resnet50-19c8e357.pth'),
    use_grid_mask=False,
    temporal_distill_cfg=None,
    use_student_history_bev=False,
    pre_lss_distill_cfg=dict(
        enable=True,
        cache_root='/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_maptrsync_p37',
        cache_suffix='.pt',
        feature_level=0,
        student_channels=256,
        teacher_channels=128,
        distill_channels=128,
        projector_hidden_channels=256,
        loss_weight=0.05,
        cosine_weight=1.0,
        l1_weight=0.25,
        use_confidence=True,
        allow_missing_cache=True,
        max_cache_items=256,
    ),
)

# The only removed pipeline operation is photometric distortion. GridMask is
# disabled above. Geometry, depth supervision, FP16, and all task losses stay
# identical to the official recipe.
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5,
         file_client_args=dict(backend='disk')),
    dict(type='CustomPointToMultiViewDepth', downsample=1,
         grid_config=dict(x=[-30.0, -30.0, 0.15], y=[-15.0, -15.0, 0.15],
                          z=[-10, 10, 20], depth=[1.0, 35.0, 0.5])),
    dict(type='PadMultiViewImageDepth', size_divisor=32),
    dict(type='DefaultFormatBundle3D', with_gt=False, with_label=False,
         class_names=['divider', 'ped_crossing', 'boundary']),
    dict(type='CustomCollect3D', keys=['img', 'gt_depth']),
]

data = dict(samples_per_gpu=4, workers_per_gpu=2,
            train=dict(pipeline=train_pipeline))
optimizer = dict(lr=3e-4)
checkpoint_config = dict(interval=2, max_keep_ckpts=12)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl', timeout=timedelta(hours=3))

