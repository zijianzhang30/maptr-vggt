_base_ = ['./maptrv2_nusc_r50_24ep_vggt_pre_lss_fp32.py']

# Use a cache re-exported with MapTR-consistent geometry:
# raw image -> fixed 0.5 resize -> pad-to-32 -> temporary pad-to-14 for VGGT
# This avoids the old square-518 preprocessing mismatch.
model = dict(
    pre_lss_distill_cfg=dict(
        cache_root='/data1/zhangzj26/maptr_data/vggt_cache_nuscenes_maptrsync_p37',
    ),
)
