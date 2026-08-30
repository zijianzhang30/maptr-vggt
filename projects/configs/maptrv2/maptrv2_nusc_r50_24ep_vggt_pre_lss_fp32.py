_base_ = ['./maptrv2_nusc_r50_24ep_vggt_pre_lss.py']

# Debug-stability variant:
# - disable fp16 to check whether mixed precision causes grad_norm NaNs
fp16 = None
