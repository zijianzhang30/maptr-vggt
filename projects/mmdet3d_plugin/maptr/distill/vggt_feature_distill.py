import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGTFeatureDistiller(nn.Module):
    """Distill frozen VGGT multi-view image features into student pre-LSS features.

    The student keeps the original MapTRv2 LSS -> BEV pipeline unchanged.
    We only supervise the multi-view image features before they enter LSS.
    """

    def __init__(self,
                 cache_root,
                 student_channels,
                 teacher_channels=None,
                 distill_channels=128,
                 projector_hidden_channels=256,
                 feature_level=0,
                 cache_suffix='.pt',
                 max_cache_items=128,
                 loss_weight=0.05,
                 cosine_weight=1.0,
                 l1_weight=0.25,
                 use_confidence=True,
                 allow_missing_cache=True):
        super().__init__()
        self.cache_root = cache_root
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        self.distill_channels = distill_channels
        self.projector_hidden_channels = projector_hidden_channels
        self.feature_level = feature_level
        self.cache_suffix = cache_suffix
        self.max_cache_items = max_cache_items
        self.loss_weight = loss_weight
        self.cosine_weight = cosine_weight
        self.l1_weight = l1_weight
        self.use_confidence = use_confidence
        self.allow_missing_cache = allow_missing_cache

        teacher_proj_in = (
            nn.LazyConv2d(projector_hidden_channels, kernel_size=1)
            if teacher_channels is None else
            nn.Conv2d(teacher_channels, projector_hidden_channels, kernel_size=1))
        self.teacher_proj = nn.Sequential(
            teacher_proj_in,
            nn.GELU(),
            nn.Conv2d(projector_hidden_channels, distill_channels, kernel_size=1))
        # The cached VGGT teacher is fixed. Keeping these parameters trainable
        # would register an intentionally detached branch with DDP.
        self.teacher_proj.requires_grad_(False)
        self.student_proj = nn.Sequential(
            nn.Conv2d(student_channels, projector_hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(projector_hidden_channels, distill_channels, kernel_size=1))

        self._cache_bank = OrderedDict()
        self._warned_missing = set()
        self._warned_corrupt = set()
        self._warned_camera_mismatch = False

    def forward(self, student_img_feats, img_metas):
        student_level = self._select_feature_level(student_img_feats)
        if student_level is None:
            return {}
        if student_level.dim() != 5:
            raise ValueError(
                f'Expected student feature shape [B, num_cam, C, H, W], got {tuple(student_level.shape)}')

        device = student_level.device
        teacher_batches = []
        student_batches = []
        weight_batches = []

        for batch_idx, img_meta in enumerate(img_metas):
            cache = self._load_cache(img_meta.get('sample_idx'))
            if cache is None:
                if self.allow_missing_cache:
                    continue
                return {}

            teacher_feat = cache.get('feature', cache.get('vggt_feat'))
            if teacher_feat is None:
                raise KeyError('VGGT cache must contain `feature` or `vggt_feat`.')
            teacher_feat = self._to_float_tensor(teacher_feat, device)

            student_feat = student_level[batch_idx]
            teacher_feat = self._align_teacher_cameras(teacher_feat, cache, img_meta)
            teacher_feat = self._resize_teacher_to_student(teacher_feat, student_feat.shape[-2:])

            num_cams = min(student_feat.shape[0], teacher_feat.shape[0])
            if num_cams == 0:
                continue
            if student_feat.shape[0] != teacher_feat.shape[0] and not self._warned_camera_mismatch:
                print('[VGGTFeatureDistiller] warning: student/teacher camera counts differ, truncating to overlap.')
                self._warned_camera_mismatch = True
            student_feat = student_feat[:num_cams]
            teacher_feat = teacher_feat[:num_cams]

            if self.use_confidence and cache.get('confidence') is not None:
                weight = self._to_float_tensor(cache['confidence'], device)
                weight = self._align_teacher_cameras(weight, cache, img_meta)
                weight = weight[:num_cams]
                weight = F.interpolate(
                    weight.unsqueeze(1),
                    size=student_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=False).squeeze(1)
                weight = weight / weight.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                weight = weight.unsqueeze(1)
            else:
                weight = torch.ones(
                    (num_cams, 1, student_feat.shape[-2], student_feat.shape[-1]),
                    device=device,
                    dtype=student_feat.dtype)

            teacher_batches.append(teacher_feat)
            student_batches.append(student_feat)
            weight_batches.append(weight)

        if not teacher_batches:
            return {'loss_vggt_img_feat': self._zero_student_loss(student_level)}

        teacher_tensor = torch.cat(teacher_batches, dim=0)
        student_tensor = torch.cat(student_batches, dim=0)
        weight_tensor = torch.cat(weight_batches, dim=0).to(dtype=student_tensor.dtype)

        with torch.no_grad():
            teacher_proj = self.teacher_proj(teacher_tensor)
        student_proj = self.student_proj(student_tensor)

        teacher_norm = F.normalize(teacher_proj, dim=1)
        student_norm = F.normalize(student_proj, dim=1)

        valid_pixels = weight_tensor.sum()
        if valid_pixels.item() < 1:
            return {'loss_vggt_img_feat': self._zero_student_loss(student_level)}

        cosine_map = 1.0 - (student_norm * teacher_norm).sum(dim=1, keepdim=True)
        loss_cos = (cosine_map * weight_tensor).sum() / (valid_pixels + 1e-6)

        l1_map = (teacher_norm - student_norm).abs().mean(dim=1, keepdim=True)
        loss_l1 = (l1_map * weight_tensor).sum() / (valid_pixels + 1e-6)

        return {
            'loss_vggt_img_feat': self.loss_weight * (
                self.cosine_weight * loss_cos + self.l1_weight * loss_l1)
        }

    def _zero_student_loss(self, student_level):
        # Preserve a stable DDP graph if a batch has no readable teacher cache.
        zero_feat = self.student_proj(student_level[:, 0, :, :1, :1])
        return zero_feat.sum() * 0.0

    def _select_feature_level(self, student_img_feats):
        if isinstance(student_img_feats, (list, tuple)):
            level = self.feature_level
            if level < 0:
                level = len(student_img_feats) + level
            if level < 0 or level >= len(student_img_feats):
                raise IndexError(
                    f'feature_level={self.feature_level} is invalid for {len(student_img_feats)} feature levels.')
            return student_img_feats[level]
        return student_img_feats

    @staticmethod
    def _to_float_tensor(data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=torch.float32)
        return torch.as_tensor(data, device=device, dtype=torch.float32)

    def _resize_teacher_to_student(self, teacher_feat, student_hw):
        if teacher_feat.shape[-2:] == student_hw:
            return teacher_feat
        return F.interpolate(
            teacher_feat,
            size=student_hw,
            mode='bilinear',
            align_corners=False)

    def _align_teacher_cameras(self, tensor, cache, img_meta):
        teacher_cam_names = cache.get('camera_names')
        student_cam_names = self._infer_student_camera_names(img_meta)
        if not teacher_cam_names or not student_cam_names:
            return tensor
        name_to_idx = {name: idx for idx, name in enumerate(teacher_cam_names)}
        aligned = []
        for cam_name in student_cam_names:
            if cam_name not in name_to_idx:
                continue
            aligned.append(tensor[name_to_idx[cam_name]])
        if not aligned:
            return tensor
        return torch.stack(aligned, dim=0)

    @staticmethod
    def _infer_student_camera_names(img_meta):
        filenames = img_meta.get('filename')
        if filenames is None:
            filenames = img_meta.get('img_filename')
        if filenames is None:
            return None
        cam_names = []
        known = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for filename in filenames:
            matched = None
            for cam_name in known:
                if cam_name in filename:
                    matched = cam_name
                    break
            if matched is None:
                return None
            cam_names.append(matched)
        return cam_names

    def _load_cache(self, sample_idx):
        if sample_idx is None:
            return None
        cache_path = os.path.join(self.cache_root, f'{sample_idx}{self.cache_suffix}')
        if cache_path in self._cache_bank:
            cache = self._cache_bank.pop(cache_path)
            self._cache_bank[cache_path] = cache
            return cache
        if not os.path.exists(cache_path):
            if cache_path not in self._warned_missing:
                print(f'[VGGTFeatureDistiller] missing cache: {cache_path}')
                self._warned_missing.add(cache_path)
            return None
        try:
            cache = torch.load(cache_path, map_location='cpu')
        except (OSError, RuntimeError) as exc:
            if cache_path not in self._warned_corrupt:
                print(f'[VGGTFeatureDistiller] failed to read cache {cache_path}: {exc}')
                self._warned_corrupt.add(cache_path)
            return None
        self._cache_bank[cache_path] = cache
        while len(self._cache_bank) > self.max_cache_items:
            self._cache_bank.popitem(last=False)
        return cache
