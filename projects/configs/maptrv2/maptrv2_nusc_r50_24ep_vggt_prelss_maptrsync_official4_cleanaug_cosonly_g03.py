_base_ = ['./maptrv2_nusc_r50_24ep_vggt_prelss_maptrsync_official4_cleanaug.py']

# Controlled comparison: keep the inherited single-frame cleanaug pipeline and
# all MapTR settings unchanged; only remove the pre-LSS VGGT feature L1 term.
model = dict(
    pre_lss_distill_cfg=dict(
        loss_weight=0.05,
        cosine_weight=1.0,
        l1_weight=0.0,
    ),
    temporal_distill_cfg=None,
)

work_dir = '/data/public/zhangzj26/maptr_vggt_cosonly_cleanaug_g03'
