import argparse
import gc
import os
import sys
from contextlib import nullcontext

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pyquaternion import Quaternion
from torchvision import transforms as TF

_ORIG_TORCH_MESHGRID = torch.meshgrid
_ORIG_INTERPOLATE = F.interpolate
try:
    _ORIG_TORCH_MESHGRID(torch.arange(1), torch.arange(1), indexing='ij')
except TypeError:
    def _meshgrid_compat(*tensors, **kwargs):
        indexing = kwargs.pop('indexing', None)
        if indexing == 'xy' and len(tensors) == 2:
            yy, xx = _ORIG_TORCH_MESHGRID(tensors[1], tensors[0], **kwargs)
            return xx, yy
        return _ORIG_TORCH_MESHGRID(*tensors, **kwargs)
    torch.meshgrid = _meshgrid_compat

try:
    _ORIG_INTERPOLATE(
        torch.zeros(1, 1, 2, 2),
        size=(4, 4),
        mode='bilinear',
        align_corners=False,
        antialias=False)
except TypeError:
    def _interpolate_compat(*args, **kwargs):
        kwargs.pop('antialias', None)
        return _ORIG_INTERPOLATE(*args, **kwargs)
    F.interpolate = _interpolate_compat


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VGGT_ROOT = os.path.join(REPO_ROOT, 'third_party', 'vggt')
if VGGT_ROOT not in sys.path:
    sys.path.insert(0, VGGT_ROOT)

from vggt.models.aggregator import Aggregator
from vggt.heads.dpt_head import DPTHead
from vggt.utils.load_fn import load_and_preprocess_images_square


CAM_ORDER = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


class VGGTFeatureDepthBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
        self.depth_head = DPTHead(
            dim_in=2048,
            output_dim=2,
            activation='exp',
            conf_activation='expp1')
        self.feature_head = DPTHead(
            dim_in=2048,
            features=128,
            feature_only=True,
            down_ratio=2,
            pos_embed=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Export VGGT cache for MapTR temporal distillation.')
    parser.add_argument(
        '--ann-files',
        nargs='+',
        required=True,
        help='One or more nuScenes temporal info pkl files.')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save per-sample cache tensors.')
    parser.add_argument(
        '--token-list-file',
        default=None,
        help='Optional file containing one sample token per line for targeted retry export.')
    parser.add_argument(
        '--model-path',
        default='/data1/zhangzj26/vggt_ckpts/model.pt',
        help='Path to local VGGT checkpoint.')
    parser.add_argument(
        '--target-size',
        type=int,
        default=518,
        help='Square image size used for the legacy square VGGT preprocessing path.')
    parser.add_argument(
        '--cache-size',
        type=int,
        default=37,
        help='Max spatial size used for saved feature / point_map / confidence tensors.')
    parser.add_argument(
        '--preprocess-mode',
        choices=['maptr_train_geom', 'square_vggt'],
        default='maptr_train_geom',
        help='Image geometry preprocessing used before VGGT inference.')
    parser.add_argument(
        '--maptr-scale',
        type=float,
        default=0.5,
        help='Deterministic image scale from the MapTR train pipeline.')
    parser.add_argument(
        '--maptr-pad-divisor',
        type=int,
        default=32,
        help='Image pad divisor from the MapTR train pipeline.')
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device string, e.g. cuda, cuda:0, cpu.')
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Global start index after concatenating infos from ann-files.')
    parser.add_argument(
        '--end-index',
        type=int,
        default=-1,
        help='Global exclusive end index, -1 means until the end.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=-1,
        help='Optional cap on number of samples to process.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing cache files.')
    parser.add_argument(
        '--save-feature-fp32',
        action='store_true',
        help='Save teacher feature maps as fp32 instead of fp16.')
    parser.add_argument(
        '--save-confidence-fp32',
        action='store_true',
        help='Save confidence maps as fp32 instead of fp16.')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=50,
        help='Print progress every N processed samples.')
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Raise on the first bad sample instead of skipping it.')
    parser.add_argument(
        '--empty-cache-per-sample',
        action='store_true',
        help='Run gc and torch.cuda.empty_cache() after each sample to reduce retry-time OOMs.')
    parser.add_argument(
        '--max-views-per-forward',
        type=int,
        default=1,
        help='Max number of camera views processed in one VGGT forward; 1 is the safest on 24GB GPUs.')
    parser.add_argument(
        '--debug-stage-log',
        action='store_true',
        help='Print detailed stage timing for model build and per-sample export.')
    return parser.parse_args()


def build_lidar2ego(info):
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(info['lidar2ego_translation'], dtype=np.float32)
    return lidar2ego


def build_camera2ego(cam_info):
    camera2ego = np.eye(4, dtype=np.float32)
    camera2ego[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
    camera2ego[:3, 3] = np.asarray(cam_info['sensor2ego_translation'], dtype=np.float32)
    return camera2ego


def resolve_image_path(image_path):
    if os.path.isabs(image_path):
        return image_path
    return os.path.abspath(os.path.join(REPO_ROOT, image_path))


def choose_cam_order(info):
    cams = info['cams']
    if all(cam_name in cams for cam_name in CAM_ORDER):
        return CAM_ORDER
    return sorted(cams.keys())


def _load_rgb_image(image_path):
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert('RGB')


def _scale_intrinsic(intrinsic, scale_x, scale_y):
    intrinsic_out = intrinsic.copy()
    intrinsic_out[0, 0] *= scale_x
    intrinsic_out[1, 1] *= scale_y
    intrinsic_out[0, 2] *= scale_x
    intrinsic_out[1, 2] *= scale_y
    return intrinsic_out


def build_processed_intrinsic(intrinsic, coord, target_size):
    x1, y1, x2, y2, width, height = coord.tolist()
    max_dim = float(max(width, height))
    scale = float(target_size) / max_dim
    pad_left = x1 / scale
    pad_top = y1 / scale

    intrinsic_proc = intrinsic.copy()
    intrinsic_proc[0, 0] *= scale
    intrinsic_proc[1, 1] *= scale
    intrinsic_proc[0, 2] = (intrinsic_proc[0, 2] + pad_left) * scale
    intrinsic_proc[1, 2] = (intrinsic_proc[1, 2] + pad_top) * scale
    return intrinsic_proc


def build_maptr_processed_intrinsic(intrinsic, scale):
    return _scale_intrinsic(intrinsic, scale, scale)


def resize_intrinsic(intrinsic, src_hw, dst_hw):
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    return _scale_intrinsic(
        intrinsic,
        float(dst_w) / float(src_w),
        float(dst_h) / float(src_h))


def _round_up(value, divisor):
    return int(np.ceil(float(value) / float(divisor)) * divisor)


def compute_cache_hw(processed_hw, cache_size):
    proc_h, proc_w = processed_hw
    if proc_h <= 0 or proc_w <= 0:
        raise ValueError(f'Invalid processed_hw={processed_hw}')
    if proc_h >= proc_w:
        cache_h = cache_size
        cache_w = max(1, int(round(cache_size * proc_w / proc_h)))
    else:
        cache_w = cache_size
        cache_h = max(1, int(round(cache_size * proc_h / proc_w)))
    return (cache_h, cache_w)


def load_and_preprocess_images_maptr(image_path_list, scale, pad_divisor, patch_size):
    """Match MapTR geometry: mmcv resize, pad-to-32, then temporary pad-to-14."""
    if len(image_path_list) == 0:
        raise ValueError('At least 1 image is required')

    images = []
    metas = []

    for image_path in image_path_list:
        img = mmcv.imread(image_path, 'color')
        height, width = img.shape[:2]

        scaled_w = int(round(width * scale))
        scaled_h = int(round(height * scale))
        img = mmcv.imresize(img, (scaled_w, scaled_h), return_scale=False)

        img = mmcv.impad_to_multiple(img, pad_divisor, pad_val=0)
        maptr_h, maptr_w = img.shape[:2]
        vggt_w = _round_up(maptr_w, patch_size)
        vggt_h = _round_up(maptr_h, patch_size)

        if vggt_h != maptr_h or vggt_w != maptr_w:
            canvas = np.zeros((vggt_h, vggt_w, 3), dtype=img.dtype)
            canvas[:maptr_h, :maptr_w] = img
        else:
            canvas = img

        canvas = mmcv.bgr2rgb(canvas)
        images.append(torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0)
        metas.append(dict(
            orig_hw=(height, width),
            scaled_hw=(scaled_h, scaled_w),
            processed_hw=(maptr_h, maptr_w),
            vggt_input_hw=(vggt_h, vggt_w),
            scale=scale))

    return torch.stack(images), metas


def depth_to_cam_points(depth_map, intrinsic):
    height, width = depth_map.shape
    grid_y = torch.arange(height, device=depth_map.device, dtype=depth_map.dtype)
    grid_x = torch.arange(width, device=depth_map.device, dtype=depth_map.dtype)
    try:
        ys, xs = torch.meshgrid(grid_y, grid_x, indexing='ij')
    except TypeError:
        ys, xs = torch.meshgrid(grid_y, grid_x)

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x_cam = (xs - cx) * depth_map / fx
    y_cam = (ys - cy) * depth_map / fy
    z_cam = depth_map
    return torch.stack([x_cam, y_cam, z_cam], dim=-1)


def camera_points_to_lidar(cam_points, camera2lidar):
    pts = cam_points.reshape(-1, 3)
    ones = torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)
    pts_h = torch.cat([pts, ones], dim=-1)
    pts_lidar = torch.matmul(pts_h, camera2lidar.t())[:, :3]
    return pts_lidar.view(*cam_points.shape[:-1], 3)


def build_amp_context(device, amp_dtype):
    if device.type != 'cuda':
        return nullcontext()
    try:
        return torch.cuda.amp.autocast(dtype=amp_dtype)
    except TypeError:
        return torch.cuda.amp.autocast()


def load_infos(ann_files):
    all_infos = []
    for ann_file in ann_files:
        data = mmcv.load(ann_file)
        infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
        all_infos.extend(infos)
    return all_infos


def load_retry_infos(all_infos, token_list_file):
    with open(token_list_file, 'r') as f:
        requested_tokens = [line.strip() for line in f if line.strip()]
    token_to_info = {info['token']: info for info in all_infos}
    retry_infos = []
    missing_tokens = []
    for token in requested_tokens:
        info = token_to_info.get(token)
        if info is None:
            missing_tokens.append(token)
        else:
            retry_infos.append(info)
    if missing_tokens:
        print(f'[VGGT cache] warning: {len(missing_tokens)} requested tokens not found in ann files')
    return retry_infos


def build_model(model_path, device):
    model = VGGTFeatureDepthBackbone()
    state_dict = torch.load(model_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    remapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('track_head.feature_extractor.'):
            remapped_state_dict[key.replace('track_head.feature_extractor.', 'feature_head.')] = value
        elif key.startswith('aggregator.') or key.startswith('depth_head.'):
            remapped_state_dict[key] = value
    missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
    if missing:
        print(f'[VGGT cache] missing keys while loading checkpoint: {len(missing)}')
    if unexpected:
        print(f'[VGGT cache] unexpected keys while loading checkpoint: {len(unexpected)}')
    for module in model.modules():
        if hasattr(module, 'fused_attn'):
            module.fused_attn = False
    model.to(device)
    model.eval()
    return model


def debug_log(args, message):
    if args.debug_stage_log:
        print(f'[VGGT cache][debug] {message}', flush=True)


def forward_views_in_chunks(model, images, device, amp_dtype, max_views_per_forward):
    num_cams = images.shape[0]
    feature_chunks = []
    depth_chunks = []
    conf_chunks = []
    patch_start_idx = None

    for start_idx in range(0, num_cams, max_views_per_forward):
        end_idx = min(start_idx + max_views_per_forward, num_cams)
        batch_images = images[start_idx:end_idx].unsqueeze(0).to(device)
        amp_ctx = build_amp_context(device, amp_dtype)
        with torch.no_grad():
            with amp_ctx:
                aggregated_tokens_list, patch_start_idx = model.aggregator(batch_images)
                feature_chunk = model.feature_head(
                    aggregated_tokens_list, batch_images, patch_start_idx)
                depth_chunk, conf_chunk = model.depth_head(
                    aggregated_tokens_list, batch_images, patch_start_idx)

        feature_chunks.append(feature_chunk.squeeze(0))
        depth_chunks.append(depth_chunk.squeeze(0).squeeze(-1))
        conf_chunks.append(conf_chunk.squeeze(0))

        del batch_images
        del aggregated_tokens_list
        del feature_chunk
        del depth_chunk
        del conf_chunk
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return (
        torch.cat(feature_chunks, dim=0),
        torch.cat(depth_chunks, dim=0),
        torch.cat(conf_chunks, dim=0),
        patch_start_idx,
    )


def export_one_sample(model, info, args, device, amp_dtype):
    sample_token = info['token']
    debug_log(args, f'sample {sample_token}: choose_cam_order')
    cam_order = choose_cam_order(info)
    image_paths = [resolve_image_path(info['cams'][cam_name]['data_path']) for cam_name in cam_order]
    debug_log(args, f'sample {sample_token}: preprocess start num_cams={len(image_paths)}')
    if args.preprocess_mode == 'maptr_train_geom':
        images, preprocess_metas = load_and_preprocess_images_maptr(
            image_paths,
            scale=args.maptr_scale,
            pad_divisor=args.maptr_pad_divisor,
            patch_size=model.aggregator.patch_size)
    else:
        images, coords = load_and_preprocess_images_square(image_paths, target_size=args.target_size)
        preprocess_metas = [
            dict(
                processed_hw=(args.target_size, args.target_size),
                scale=None,
                coord=coord)
            for coord in coords
        ]

    num_cams = len(cam_order)
    debug_log(args, f'sample {sample_token}: forward_views_in_chunks start')
    feature_map, depth_map, depth_conf, patch_start_idx = forward_views_in_chunks(
        model,
        images,
        device,
        amp_dtype,
        max(1, args.max_views_per_forward))
    debug_log(args, f'sample {sample_token}: forward_views_in_chunks done feature_shape={tuple(feature_map.shape)}')

    if args.preprocess_mode == 'maptr_train_geom':
        processed_hw = preprocess_metas[0]['processed_hw']
        proc_h, proc_w = processed_hw
        feature_map = feature_map[..., :proc_h, :proc_w]
        depth_map = depth_map[..., :proc_h, :proc_w]
        depth_conf = depth_conf[..., :proc_h, :proc_w]
        cache_hw = compute_cache_hw(processed_hw, args.cache_size)
    else:
        cache_hw = (args.cache_size, args.cache_size)

    debug_log(args, f'sample {sample_token}: resize-to-cache start cache_hw={cache_hw}')
    feature_map = F.adaptive_avg_pool2d(feature_map, output_size=cache_hw)
    depth_small = F.interpolate(
        depth_map.unsqueeze(1),
        size=cache_hw,
        mode='bilinear',
        align_corners=False).squeeze(1)
    conf_small = F.interpolate(
        depth_conf.unsqueeze(1),
        size=cache_hw,
        mode='bilinear',
        align_corners=False).squeeze(1)

    lidar2ego = build_lidar2ego(info)
    ego2lidar = np.linalg.inv(lidar2ego).astype(np.float32)

    debug_log(args, f'sample {sample_token}: point_map build start')
    point_maps = []
    for cam_idx, cam_name in enumerate(cam_order):
        cam_info = info['cams'][cam_name]
        intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32)
        if args.preprocess_mode == 'maptr_train_geom':
            intrinsic_proc = build_maptr_processed_intrinsic(
                intrinsic, preprocess_metas[cam_idx]['scale'])
            intrinsic_cache = resize_intrinsic(
                intrinsic_proc, preprocess_metas[cam_idx]['processed_hw'], cache_hw)
        else:
            intrinsic_proc = build_processed_intrinsic(
                intrinsic, preprocess_metas[cam_idx]['coord'], args.target_size)
            intrinsic_cache = resize_intrinsic(
                intrinsic_proc, (args.target_size, args.target_size), cache_hw)

        camera2ego = build_camera2ego(cam_info)
        camera2lidar = ego2lidar @ camera2ego

        intrinsic_cache = torch.from_numpy(intrinsic_cache).to(device=device, dtype=depth_small.dtype)
        camera2lidar = torch.from_numpy(camera2lidar).to(device=device, dtype=depth_small.dtype)

        cam_points = depth_to_cam_points(depth_small[cam_idx], intrinsic_cache)
        lidar_points = camera_points_to_lidar(cam_points, camera2lidar)
        point_maps.append(lidar_points)

    point_maps = torch.stack(point_maps, dim=0)
    debug_log(args, f'sample {sample_token}: point_map build done point_shape={tuple(point_maps.shape)}')

    feature_dtype = torch.float32 if args.save_feature_fp32 else torch.float16
    conf_dtype = torch.float32 if args.save_confidence_fp32 else torch.float16
    cache = dict(
        feature=feature_map.detach().cpu().to(feature_dtype),
        point_map=point_maps.detach().cpu().to(torch.float32),
        confidence=conf_small.detach().cpu().to(conf_dtype),
        camera_names=cam_order,
        source_hw=cache_hw,
        preprocess_mode=args.preprocess_mode,
        sample_idx=info['token'],
    )
    return cache


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    requested_device = torch.device(args.device)
    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        print('[VGGT cache] CUDA is not available, falling back to CPU.')
        device = torch.device('cpu')
    else:
        device = requested_device

    amp_dtype = torch.bfloat16
    if device.type == 'cuda':
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        if torch.cuda.get_device_capability(device_index)[0] < 8:
            amp_dtype = torch.float16

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    infos = load_infos(args.ann_files)
    if args.token_list_file is not None:
        infos = load_retry_infos(infos, args.token_list_file)
        total_infos = len(infos)
    else:
        total_infos = len(infos)
        end_index = total_infos if args.end_index < 0 else min(args.end_index, total_infos)
        infos = infos[args.start_index:end_index]
        if args.max_samples > 0:
            infos = infos[:args.max_samples]

    print(f'[VGGT cache] device={device}, samples={len(infos)}, output_dir={args.output_dir}')
    debug_log(args, f'build_model start model_path={args.model_path}')
    model = build_model(args.model_path, device)
    debug_log(args, 'build_model done')

    processed = 0
    skipped = 0
    failed = 0
    for local_idx, info in enumerate(infos):
        sample_idx = info['token']
        output_path = os.path.join(args.output_dir, f'{sample_idx}.pt')
        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue

        try:
            debug_log(args, f'local_idx={local_idx} sample={sample_idx}: export_one_sample start')
            cache = export_one_sample(model, info, args, device, amp_dtype)
            debug_log(args, f'local_idx={local_idx} sample={sample_idx}: export_one_sample done; saving')
            tmp_path = output_path + '.tmp'
            torch.save(cache, tmp_path)
            os.replace(tmp_path, output_path)
            processed += 1
            if args.log_interval > 0 and processed % args.log_interval == 0:
                print(
                    f'[VGGT cache] processed={processed} skipped={skipped} failed={failed} '
                    f'last_sample={sample_idx} global_idx={args.start_index + local_idx}')
        except Exception as exc:
            failed += 1
            print(f'[VGGT cache] failed sample {sample_idx}: {exc}')
            if args.fail_fast:
                raise
        finally:
            if args.empty_cache_per_sample:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    print(f'[VGGT cache] done processed={processed} skipped={skipped} failed={failed}')


if __name__ == '__main__':
    main()
