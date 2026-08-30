import os
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyquaternion import Quaternion


class TemporalVGGTBEVDistiller(nn.Module):
    """History-only VGGT -> BEV teacher distillation for MapTR/MapTRv2.

    Expected cache format per sample token:
        {
            'feature' or 'vggt_feat': Tensor[num_cams, C, H, W],
            'point_map': Tensor[num_cams, 3, H, W] or [num_cams, H, W, 3],
            'confidence': Tensor[num_cams, H, W], optional,
        }
    """

    def __init__(self,
                 cache_root,
                 pc_range,
                 bev_h,
                 bev_w,
                 student_channels,
                 teacher_channels=None,
                 distill_channels=128,
                 projector_hidden_channels=256,
                 teacher_num_frames=3,
                 ann_file=None,
                 use_map_mask=True,
                 use_confidence=True,
                 ground_height_range=None,
                 cache_suffix='.pt',
                 max_cache_items=32,
                 loss_weight=0.05,
                 cosine_weight=1.0,
                 l1_weight=0.25,
                 allow_missing_cache=True):
        super().__init__()
        self.cache_root = cache_root
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        self.distill_channels = distill_channels
        self.projector_hidden_channels = projector_hidden_channels
        self.teacher_num_frames = teacher_num_frames
        self.ann_file = ann_file
        self.use_map_mask = use_map_mask
        self.use_confidence = use_confidence
        self.ground_height_range = ground_height_range
        self.cache_suffix = cache_suffix
        self.max_cache_items = max_cache_items
        self.loss_weight = loss_weight
        self.cosine_weight = cosine_weight
        self.l1_weight = l1_weight
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
        self._token_meta_index = self._build_token_meta_index(ann_file)

    def forward(self, student_bev, queue_img_metas, gt_seg_mask=None):
        teacher_bev, teacher_valid, valid_batch_idx = self.build_teacher_bev(
            queue_img_metas, student_bev.device)
        if teacher_bev is None:
            return {'loss_vggt_feat': self._zero_student_loss(student_bev)}

        student_bev = student_bev.index_select(0, valid_batch_idx)
        if gt_seg_mask is not None:
            gt_seg_mask = self._select_batch_items(
                gt_seg_mask, valid_batch_idx, device=student_bev.device)

        if teacher_bev.shape[-2:] != student_bev.shape[-2:]:
            teacher_bev = F.interpolate(
                teacher_bev, size=student_bev.shape[-2:], mode='bilinear', align_corners=False)
            teacher_valid = F.interpolate(
                teacher_valid, size=student_bev.shape[-2:], mode='nearest')

        distill_mask = teacher_valid
        if self.use_map_mask and gt_seg_mask is not None:
            if gt_seg_mask.dim() == 3:
                gt_seg_mask = gt_seg_mask.unsqueeze(1)
            map_mask = (gt_seg_mask.sum(dim=1, keepdim=True) > 0).float().to(student_bev.device)
            if map_mask.shape[-2:] != student_bev.shape[-2:]:
                map_mask = F.interpolate(map_mask, size=student_bev.shape[-2:], mode='nearest')
            distill_mask = distill_mask * map_mask

        valid_pixels = distill_mask.sum()
        if valid_pixels.item() < 1:
            return {'loss_vggt_feat': self._zero_student_loss(student_bev)}

        with torch.no_grad():
            teacher_proj = self.teacher_proj(teacher_bev)
        student_proj = self.student_proj(student_bev)

        teacher_norm = F.normalize(teacher_proj, dim=1)
        student_norm = F.normalize(student_proj, dim=1)

        cosine_map = 1.0 - (student_norm * teacher_norm).sum(dim=1, keepdim=True)
        loss_cos = (cosine_map * distill_mask).sum() / (valid_pixels + 1e-6)

        l1_map = (teacher_norm - student_norm).abs().mean(dim=1, keepdim=True)
        loss_l1 = (l1_map * distill_mask).sum() / (valid_pixels + 1e-6)

        return {
            'loss_vggt_feat': self.loss_weight * (self.cosine_weight * loss_cos + self.l1_weight * loss_l1)
        }

    def _zero_student_loss(self, student_bev):
        # Keep the student projector in the autograd/DDP graph even when this
        # batch has no valid temporal teacher supervision.
        zero_feat = self.student_proj(student_bev[..., :1, :1])
        return zero_feat.sum() * 0.0

    def build_teacher_bev(self, batch_queue_metas, device):
        teacher_bevs = []
        teacher_valids = []
        valid_batch_idx = []

        for batch_idx, sample_queue_metas in enumerate(batch_queue_metas):
            selected_metas = self._select_teacher_frame_metas(sample_queue_metas)
            if not selected_metas:
                continue
            current_meta = selected_metas[-1]
            current_lidar2global = self._to_tensor(current_meta['lidar2global'], device=device)

            fused_bev = None
            fused_valid = None
            fused_hits = None
            used_frame_count = 0
            sample_invalid = False
            for frame_meta in selected_metas:
                cache = self._load_cache(frame_meta['sample_idx'])
                if cache is None:
                    if not self.allow_missing_cache:
                        sample_invalid = True
                        break
                    continue

                feature = cache.get('feature', cache.get('vggt_feat'))
                point_map = cache.get('point_map')
                confidence = cache.get('confidence')
                if feature is None or point_map is None:
                    raise KeyError(
                        f"VGGT cache for {frame_meta['sample_idx']} must contain feature/vggt_feat and point_map.")

                src_lidar2global = self._to_tensor(frame_meta['lidar2global'], device=device)
                src_to_tgt = torch.linalg.inv(current_lidar2global) @ src_lidar2global

                bev_feat, bev_valid, bev_hits = self.point_to_bev(
                    feature=feature,
                    point_map=point_map,
                    confidence=confidence,
                    src_to_tgt=src_to_tgt,
                    device=device)

                if bev_feat is None:
                    continue
                used_frame_count += 1
                if fused_bev is None:
                    fused_bev = bev_feat
                    fused_valid = bev_valid
                    fused_hits = bev_hits
                else:
                    fused_bev = fused_bev + bev_feat
                    fused_valid = fused_valid + bev_valid
                    fused_hits = fused_hits + bev_hits

            if sample_invalid:
                continue
            if fused_bev is None:
                continue

            teacher_support = torch.clamp(fused_hits / max(used_frame_count, 1), min=0.0, max=1.0)
            teacher_valid = (fused_valid > 0).float() * teacher_support
            teacher_bev = fused_bev / (fused_valid + 1e-6)
            teacher_bevs.append(teacher_bev)
            teacher_valids.append(teacher_valid)
            valid_batch_idx.append(batch_idx)

        if not teacher_bevs:
            return None, None, None
        teacher_bevs = torch.stack(teacher_bevs, dim=0)
        teacher_valids = torch.stack(teacher_valids, dim=0)
        valid_batch_idx = torch.tensor(valid_batch_idx, device=device, dtype=torch.long)
        return teacher_bevs, teacher_valids, valid_batch_idx

    def point_to_bev(self, feature, point_map, confidence, src_to_tgt, device):
        feature = self._to_tensor(feature, device=device, dtype=torch.float32)
        point_map = self._to_tensor(point_map, device=device, dtype=torch.float32)
        if confidence is not None:
            confidence = self._to_tensor(confidence, device=device, dtype=torch.float32)

        if point_map.dim() != 4:
            raise ValueError(f'Expected point_map dim=4, got shape={tuple(point_map.shape)}')
        if point_map.shape[1] != 3 and point_map.shape[-1] == 3:
            point_map = point_map.permute(0, 3, 1, 2).contiguous()
        if point_map.shape[1] != 3:
            raise ValueError(f'Expected point_map channel=3, got shape={tuple(point_map.shape)}')

        if feature.shape[-2:] != point_map.shape[-2:]:
            feature = F.interpolate(feature, size=point_map.shape[-2:], mode='bilinear', align_corners=False)
        if confidence is not None:
            if confidence.dim() == 4 and confidence.shape[1] == 1:
                confidence = confidence[:, 0]
            if confidence.shape[-2:] != point_map.shape[-2:]:
                confidence = F.interpolate(
                    confidence.unsqueeze(1), size=point_map.shape[-2:], mode='nearest').squeeze(1)

        num_cams, feat_c, feat_h, feat_w = feature.shape
        points = point_map.permute(0, 2, 3, 1).reshape(-1, 3)
        ones = torch.ones((points.shape[0], 1), device=device, dtype=points.dtype)
        points_h = torch.cat([points, ones], dim=1)
        warped = (src_to_tgt @ points_h.t()).t()[:, :3]

        feats = feature.permute(0, 2, 3, 1).reshape(-1, feat_c)
        if confidence is None or not self.use_confidence:
            weights = torch.ones((warped.shape[0],), device=device, dtype=feats.dtype)
        else:
            weights = confidence.reshape(-1).to(feats.dtype)

        finite_mask = torch.isfinite(warped).all(dim=1) & torch.isfinite(feats).all(dim=1) & torch.isfinite(weights)
        x = warped[:, 0]
        y = warped[:, 1]
        z = warped[:, 2]
        range_mask = (
            (x >= self.pc_range[0]) & (x < self.pc_range[3]) &
            (y >= self.pc_range[1]) & (y < self.pc_range[4]))
        valid = finite_mask & range_mask & (weights > 0)
        if self.ground_height_range is not None:
            z_min, z_max = self.ground_height_range
            valid = valid & (z >= z_min) & (z <= z_max)
        if valid.sum().item() == 0:
            return None, None, None

        x = x[valid]
        y = y[valid]
        feats = feats[valid]
        weights = weights[valid]

        bx = torch.clamp(
            ((x - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0]) * self.bev_w).long(),
            min=0,
            max=self.bev_w - 1)
        by = torch.clamp(
            ((y - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1]) * self.bev_h).long(),
            min=0,
            max=self.bev_h - 1)
        linear_idx = by * self.bev_w + bx

        bev_sum = feats.new_zeros((self.bev_h * self.bev_w, feat_c))
        weight_sum = feats.new_zeros((self.bev_h * self.bev_w,))
        hit_sum = feats.new_zeros((self.bev_h * self.bev_w,))
        bev_sum.index_add_(0, linear_idx, feats * weights.unsqueeze(1))
        weight_sum.index_add_(0, linear_idx, weights)
        hit_sum.index_add_(0, linear_idx, torch.ones_like(weights))

        bev_feat = bev_sum.t().reshape(feat_c, self.bev_h, self.bev_w)
        bev_valid = weight_sum.reshape(1, self.bev_h, self.bev_w)
        bev_hits = hit_sum.reshape(1, self.bev_h, self.bev_w)
        return bev_feat, bev_valid, bev_hits

    def _select_teacher_frame_metas(self, sample_queue_metas):
        frame_ids = sorted(sample_queue_metas.keys())
        if not frame_ids:
            return []
        current_meta = sample_queue_metas[frame_ids[-1]]
        current_token = current_meta.get('sample_idx')
        if current_token is not None and self._token_meta_index:
            traced_metas = []
            token = current_token
            while token and len(traced_metas) < self.teacher_num_frames:
                traced_meta = self._token_meta_index.get(token)
                if traced_meta is None:
                    break
                traced_metas.append(traced_meta)
                token = traced_meta.get('prev_idx')
            if traced_metas:
                return list(reversed(traced_metas))
        return [sample_queue_metas[frame_id] for frame_id in frame_ids[-self.teacher_num_frames:]]

    def _build_token_meta_index(self, ann_file):
        if ann_file is None:
            return {}
        ann_files = ann_file if isinstance(ann_file, (list, tuple)) else [ann_file]
        token_meta_index = {}
        for ann_path in ann_files:
            data = mmcv.load(ann_path)
            infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
            for info in infos:
                token_meta_index[info['token']] = dict(
                    sample_idx=info['token'],
                    prev_idx=info.get('prev'),
                    lidar2global=self._compute_lidar2global(info))
        return token_meta_index

    @staticmethod
    def _compute_lidar2global(info):
        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = np.asarray(info['lidar2ego_translation'], dtype=np.float32)
        ego2global = np.eye(4, dtype=np.float32)
        ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = np.asarray(info['ego2global_translation'], dtype=np.float32)
        return ego2global @ lidar2ego

    def _load_cache(self, sample_idx):
        cache_path = os.path.join(self.cache_root, f'{sample_idx}{self.cache_suffix}')
        if cache_path in self._cache_bank:
            cache = self._cache_bank.pop(cache_path)
            self._cache_bank[cache_path] = cache
            return cache

        if not os.path.exists(cache_path):
            if cache_path not in self._warned_missing:
                print(f'[TemporalVGGTBEVDistiller] missing cache: {cache_path}')
                self._warned_missing.add(cache_path)
            return None

        try:
            cache = torch.load(cache_path, map_location='cpu')
        except (OSError, RuntimeError) as exc:
            if cache_path not in self._warned_missing:
                print(f'[TemporalVGGTBEVDistiller] failed to read cache: {cache_path} ({exc})')
                self._warned_missing.add(cache_path)
            return None
        self._cache_bank[cache_path] = cache
        while len(self._cache_bank) > self.max_cache_items:
            self._cache_bank.popitem(last=False)
        return cache

    @staticmethod
    def _to_tensor(data, device, dtype=None):
        if isinstance(data, torch.Tensor):
            tensor = data.to(device=device)
            return tensor if dtype is None else tensor.to(dtype=dtype)
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(device=device)
            return tensor if dtype is None else tensor.to(dtype=dtype)
        tensor = torch.tensor(data, device=device)
        return tensor if dtype is None else tensor.to(dtype=dtype)

    @staticmethod
    def _select_batch_items(batch_data, valid_batch_idx, device):
        if isinstance(batch_data, torch.Tensor):
            return batch_data.index_select(0, valid_batch_idx.to(batch_data.device))
        if isinstance(batch_data, (list, tuple)):
            selected = [batch_data[idx] for idx in valid_batch_idx.tolist()]
            if not selected:
                return None
            if isinstance(selected[0], torch.Tensor):
                return torch.stack([item.to(device=device) for item in selected], dim=0)
            return selected
        return batch_data
