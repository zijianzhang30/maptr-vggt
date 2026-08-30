from .nuscenes_dataset import CustomNuScenesDataset
from .builder import custom_build_dataset

from .nuscenes_map_dataset import CustomNuScenesLocalMapDataset
from .nuscenes_offlinemap_dataset import CustomNuScenesOfflineLocalMapDataset

try:
    from .av2_map_dataset import CustomAV2LocalMapDataset
    from .av2_offlinemap_dataset import CustomAV2OfflineLocalMapDataset
except Exception:
    # AV2 tooling pulls in newer Python/type-hint expectations; keep nuScenes
    # imports usable even when AV2 dependencies are not aligned yet.
    CustomAV2LocalMapDataset = None
    CustomAV2OfflineLocalMapDataset = None
__all__ = [
    'CustomNuScenesDataset','CustomNuScenesLocalMapDataset'
]
