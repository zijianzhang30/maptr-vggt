_base_ = ['./maptrv2_nusc_r50_24ep.py']

# A2 minimal version:
# - history-only 3-frame VGGT teacher
# - current-frame student only
# - REPA-style normalized projector alignment in BEV space

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        queue_length=3,
    ),
)

model = dict(
    pretrained=dict(
        img='/home/zhangzj26/maptr_ckpts/resnet50-19c8e357.pth',
    ),
    use_student_history_bev=False,
    temporal_distill_cfg=dict(
        enable=True,
        cache_root='/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_p37',
        ann_file='/home/zhangzj26/MapTR/data/nuscenes/nuscenes_map_infos_temporal_train.pkl',
        cache_suffix='.pt',
        teacher_num_frames=3,
        teacher_channels=128,
        distill_channels=128,
        projector_hidden_channels=256,
        loss_weight=0.05,
        cosine_weight=1.0,
        l1_weight=0.25,
        use_map_mask=True,
        use_confidence=True,
        ground_height_range=(-3.0, 1.0),
        allow_missing_cache=True,
        max_cache_items=64,
    ),
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ],
)
