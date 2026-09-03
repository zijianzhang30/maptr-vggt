_base_ = ['./maptrv2_nusc_r50_24ep_vggt_prelss_maptrsync_official4_cleanaug.py']

# Strict controlled comparison: only replace Cos+L1 with cosine-only.
model = dict(
    pre_lss_distill_cfg=dict(
        loss_weight=0.05,
        cosine_weight=1.0,
        l1_weight=0.0,
    ),
    temporal_distill_cfg=None,
)

work_dir = '/home/zhangzj26/MapTR/work_dirs/maptrv2_vggt_prelss_maptrsync_official4_cleanaug_cosonly'
