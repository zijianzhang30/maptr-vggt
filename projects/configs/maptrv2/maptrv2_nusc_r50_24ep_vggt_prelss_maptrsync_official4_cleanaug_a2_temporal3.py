from datetime import timedelta

_base_ = [
    './maptrv2_nusc_r50_24ep_vggt_prelss_maptrsync_official4_cleanaug.py'
]

_cache_root_ = '/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_maptrsync_p37'
_train_ann_ = '/home/zhangzj26/MapTR/data/nuscenes/nuscenes_map_infos_temporal_train.pkl'

model = dict(
    # Keep the student current-frame only. Temporal information is privileged
    # teacher supervision during training, not an inference-time input.
    use_student_history_bev=False,
    # LSSTransform never reads the BEVFormer-only embeddings below. Freeze
    # them so static DDP does not wait for gradients they cannot receive.
    freeze_lss_unused_transformer_params=True,
    temporal_distill_cfg=dict(
        _delete_=True,
        enable=True,
        cache_root=_cache_root_,
        ann_file=_train_ann_,
        teacher_num_frames=3,
        student_channels=256,
        teacher_channels=128,
        distill_channels=128,
        projector_hidden_channels=256,
        use_map_mask=True,
        use_confidence=True,
        ground_height_range=None,
        cache_suffix='.pt',
        max_cache_items=32,
        loss_weight=0.05,
        cosine_weight=1.0,
        l1_weight=0.25,
        allow_missing_cache=True,
    ),
)

data = dict(
    samples_per_gpu=2,
    # A 3-frame sample is too large for worker shared-memory transfer here.
    workers_per_gpu=4,
    train=dict(queue_length=3),
)

optimizer = dict(lr=1.5e-4)
checkpoint_config = dict(interval=2, max_keep_ckpts=12)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl', timeout=timedelta(hours=3))

# Both ranks emit the same complete loss set. Disable DDP's unused-parameter
# graph traversal so a real dynamic branch fails fast with the parameter name.
find_unused_parameters = False
