_base_ = ['./maptrv2_nusc_r50_24ep.py']

# Pre-LSS VGGT distillation:
# - keep original MapTRv2 LSS -> BEV unchanged
# - distill frozen VGGT multi-view features into student img_neck features
# - avoid temporal teacher BEV construction and heavy history cache I/O

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)

model = dict(
    pretrained=dict(
        img='/home/zhangzj26/maptr_ckpts/resnet50-19c8e357.pth',
    ),
    temporal_distill_cfg=None,
    use_student_history_bev=False,
    pre_lss_distill_cfg=dict(
        enable=True,
        cache_root='/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_p37',
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

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ],
)
