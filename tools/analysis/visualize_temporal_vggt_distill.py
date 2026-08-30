import argparse
import importlib
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config


REPO_ROOT = Path(__file__).resolve().parents[2]
MMDET3D_ROOT = REPO_ROOT / "mmdetection3d"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MMDET3D_ROOT) not in sys.path:
    sys.path.insert(0, str(MMDET3D_ROOT))

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.maptr.distill.temporal_vggt_bev_distill import (  # noqa: E402
    TemporalVGGTBEVDistiller,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Temporal VGGT distillation masks for MapTRv2 cases."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "projects/configs/maptrv2/maptrv2_nusc_r50_24ep_vggt_a2.py"),
        help="MapTRv2 config file.",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        required=True,
        help="Sample tokens to visualize.",
    )
    parser.add_argument(
        "--ann-file",
        default=None,
        help="Override dataset/distiller ann file. Defaults to cfg.data.train.ann_file.",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Override VGGT cache root. Defaults to cfg.model.temporal_distill_cfg.cache_root.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save exported figures and summary json.",
    )
    parser.add_argument(
        "--queue-length",
        type=int,
        default=None,
        help="Dataset queue length used for prepare_train_data. Defaults to teacher_num_frames.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for tensor ops. cpu is enough for visualization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset temporal augmentation path.",
    )
    return parser.parse_args()


def import_plugins(cfg):
    if not getattr(cfg, "plugin", False):
        return
    plugin_dir = getattr(cfg, "plugin_dir", None)
    if plugin_dir is None:
        return
    module_parts = os.path.dirname(plugin_dir).split("/")
    module_path = module_parts[0]
    for part in module_parts[1:]:
        module_path += "." + part
    importlib.import_module(module_path)


def build_train_like_dataset(cfg, ann_file, queue_length, seed):
    dataset_cfg = deepcopy(cfg.data.train)
    dataset_cfg.ann_file = ann_file
    dataset_cfg.queue_length = queue_length
    dataset_cfg.test_mode = False
    random.seed(seed)
    np.random.seed(seed)
    return build_dataset(dataset_cfg)


def build_distiller(cfg, ann_file, cache_root, ground_height_range, device):
    distill_cfg = cfg.model.temporal_distill_cfg
    return TemporalVGGTBEVDistiller(
        cache_root=cache_root,
        ann_file=ann_file,
        cache_suffix=distill_cfg.get("cache_suffix", ".pt"),
        teacher_num_frames=distill_cfg.teacher_num_frames,
        teacher_channels=distill_cfg.teacher_channels,
        distill_channels=distill_cfg.distill_channels,
        projector_hidden_channels=distill_cfg.projector_hidden_channels,
        pc_range=cfg.point_cloud_range,
        bev_h=cfg.bev_h_,
        bev_w=cfg.bev_w_,
        student_channels=cfg._dim_,
        use_map_mask=distill_cfg.get("use_map_mask", True),
        use_confidence=distill_cfg.get("use_confidence", True),
        ground_height_range=ground_height_range,
        max_cache_items=distill_cfg.get("max_cache_items", 64),
        allow_missing_cache=distill_cfg.get("allow_missing_cache", True),
    ).to(device)


def load_case_example(dataset, token):
    token_to_index = {info["token"]: idx for idx, info in enumerate(dataset.data_infos)}
    if token not in token_to_index:
        raise KeyError(f"Token {token} not found in dataset ann file.")
    index = token_to_index[token]
    example = dataset.prepare_train_data(index)
    if example is None:
        raise RuntimeError(f"Dataset returned None for token {token} at index {index}.")
    return example, index


def fuse_teacher_case(distiller, sample_queue_metas, device):
    selected_metas = distiller._select_teacher_frame_metas(sample_queue_metas)
    if not selected_metas:
        raise RuntimeError("No teacher frames were selected for this sample.")

    current_meta = selected_metas[-1]
    current_lidar2global = distiller._to_tensor(
        current_meta["lidar2global"], device=device
    )

    fused_bev = None
    fused_valid = None
    fused_hits = None
    used_frame_count = 0
    used_tokens = []
    missing_tokens = []

    for frame_meta in selected_metas:
        sample_idx = frame_meta["sample_idx"]
        cache = distiller._load_cache(sample_idx)
        if cache is None:
            missing_tokens.append(sample_idx)
            continue

        src_lidar2global = distiller._to_tensor(frame_meta["lidar2global"], device=device)
        src_to_tgt = torch.linalg.inv(current_lidar2global) @ src_lidar2global

        bev_feat, bev_valid, bev_hits = distiller.point_to_bev(
            feature=cache.get("feature", cache.get("vggt_feat")),
            point_map=cache.get("point_map"),
            confidence=cache.get("confidence"),
            src_to_tgt=src_to_tgt,
            device=device,
        )
        if bev_feat is None:
            continue

        used_frame_count += 1
        used_tokens.append(sample_idx)
        if fused_bev is None:
            fused_bev = bev_feat
            fused_valid = bev_valid
            fused_hits = bev_hits
        else:
            fused_bev = fused_bev + bev_feat
            fused_valid = fused_valid + bev_valid
            fused_hits = fused_hits + bev_hits

    if fused_bev is None:
        raise RuntimeError(
            "Teacher BEV is empty. All frames were missing cache or filtered out."
        )

    result = {
        "selected_tokens": [meta["sample_idx"] for meta in selected_metas],
        "used_tokens": used_tokens,
        "missing_tokens": missing_tokens,
        "used_frame_count": used_frame_count,
        "teacher_bev": fused_bev / (fused_valid + 1e-6),
        "fused_valid": fused_valid,
        "fused_hits": fused_hits,
        "binary_valid": (fused_valid > 0).float(),
    }
    return result


def to_numpy(tensor):
    return tensor.detach().cpu().float().numpy()


def compute_case_outputs(
    example,
    dist_no_ground,
    dist_ground,
    use_map_mask,
    device,
):
    sample_queue_metas = example["img_metas"].data
    gt_seg_mask = example["gt_seg_mask"].data.to(device=device)
    if gt_seg_mask.dim() == 2:
        gt_seg_mask = gt_seg_mask.unsqueeze(0)

    before = fuse_teacher_case(dist_no_ground, sample_queue_metas, device)
    after = fuse_teacher_case(dist_ground, sample_queue_metas, device)

    if before["selected_tokens"] != after["selected_tokens"]:
        raise RuntimeError("Teacher frame selection mismatch between debug branches.")

    map_mask = (gt_seg_mask.sum(dim=0, keepdim=True) > 0).float()
    teacher_support = torch.clamp(
        after["fused_hits"] / max(after["used_frame_count"], 1), min=0.0, max=1.0
    )
    teacher_valid_weighted = after["binary_valid"] * teacher_support
    distill_mask = teacher_valid_weighted * map_mask if use_map_mask else teacher_valid_weighted
    removed_by_ground = before["binary_valid"] * (1.0 - after["binary_valid"])
    removed_outside_map = removed_by_ground * (1.0 - map_mask)

    before_binary = before["binary_valid"] > 0
    after_binary = after["binary_valid"] > 0
    map_binary = map_mask > 0
    removed_binary = removed_by_ground > 0
    distill_binary = distill_mask > 0

    def ratio(numerator, denominator):
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    before_cells = int(before_binary.sum().item())
    after_cells = int(after_binary.sum().item())
    removed_cells = int(removed_binary.sum().item())
    map_cells = int(map_binary.sum().item())
    distill_cells = int(distill_binary.sum().item())
    before_on_map = int((before_binary & map_binary).sum().item())
    after_on_map = int((after_binary & map_binary).sum().item())
    removed_outside_cells = int((removed_binary & (~map_binary)).sum().item())
    one_frame_cells = int(((teacher_support > 0) & (teacher_support <= 0.34) & after_binary).sum().item())
    two_frame_cells = int(((teacher_support > 0.34) & (teacher_support <= 0.67) & after_binary).sum().item())
    three_frame_cells = int(((teacher_support > 0.67) & after_binary).sum().item())

    summary = {
        "selected_tokens": before["selected_tokens"],
        "used_tokens_before_filter": before["used_tokens"],
        "used_tokens_after_filter": after["used_tokens"],
        "missing_tokens_before_filter": before["missing_tokens"],
        "missing_tokens_after_filter": after["missing_tokens"],
        "used_frame_count_before_filter": before["used_frame_count"],
        "used_frame_count_after_filter": after["used_frame_count"],
        "before_valid_cells": before_cells,
        "after_valid_cells": after_cells,
        "removed_by_ground_cells": removed_cells,
        "removed_outside_map_cells": removed_outside_cells,
        "map_mask_cells": map_cells,
        "distill_mask_cells": distill_cells,
        "before_on_map_ratio": ratio(before_on_map, before_cells),
        "after_on_map_ratio": ratio(after_on_map, after_cells),
        "removed_ratio": ratio(removed_cells, before_cells),
        "removed_outside_ratio": ratio(removed_outside_cells, removed_cells),
        "teacher_support_mean_on_valid": ratio(
            float((teacher_support * after["binary_valid"]).sum().item()), after_cells
        ),
        "teacher_support_p90_on_valid": float(
            torch.quantile(teacher_support[after_binary], 0.9).item()
        ) if after_cells > 0 else 0.0,
        "support_1frame_cells": one_frame_cells,
        "support_2frame_cells": two_frame_cells,
        "support_3frame_cells": three_frame_cells,
    }

    outputs = {
        "before_valid": to_numpy(before["binary_valid"][0]),
        "removed_by_ground": to_numpy(removed_by_ground[0]),
        "removed_outside_map": to_numpy(removed_outside_map[0]),
        "after_valid": to_numpy(after["binary_valid"][0]),
        "teacher_support": to_numpy(teacher_support[0]),
        "teacher_valid_weighted": to_numpy(teacher_valid_weighted[0]),
        "map_mask": to_numpy(map_mask[0]),
        "distill_mask": to_numpy(distill_mask[0]),
        "summary": summary,
    }
    return outputs


def add_panel(ax, image, title, cmap="viridis", vmin=None, vmax=None):
    ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def save_case_figure(token, outputs, out_path):
    summary = outputs["summary"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    panels = [
        ("before_valid", "Before z-filter", "gray", 0.0, 1.0),
        ("removed_by_ground", "Removed by z-filter", "magma", 0.0, 1.0),
        ("removed_outside_map", "Removed outside map", "magma", 0.0, 1.0),
        ("after_valid", "After z-filter", "gray", 0.0, 1.0),
        ("teacher_support", "Support ratio", "viridis", 0.0, 1.0),
        ("teacher_valid_weighted", "Weighted teacher_valid", "viridis", 0.0, 1.0),
        ("map_mask", "Map mask", "gray", 0.0, 1.0),
        ("distill_mask", "Final distill_mask", "viridis", 0.0, 1.0),
    ]

    for ax, (key, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        add_panel(ax, outputs[key], title, cmap=cmap, vmin=vmin, vmax=vmax)

    fig.suptitle(
        "\n".join(
            [
                f"{token}",
                (
                    f"frames={summary['used_frame_count_after_filter']} | "
                    f"before={summary['before_valid_cells']} after={summary['after_valid_cells']} "
                    f"removed={summary['removed_by_ground_cells']} "
                    f"({summary['removed_ratio'] * 100:.1f}%)"
                ),
                (
                    f"on-map overlap {summary['before_on_map_ratio'] * 100:.1f}% -> "
                    f"{summary['after_on_map_ratio'] * 100:.1f}% | "
                    f"removed outside map {summary['removed_outside_ratio'] * 100:.1f}% | "
                    f"support 1f/2f/3f = "
                    f"{summary['support_1frame_cells']}/"
                    f"{summary['support_2frame_cells']}/"
                    f"{summary['support_3frame_cells']}"
                ),
            ]
        ),
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugins(cfg)

    distill_cfg = cfg.model.temporal_distill_cfg
    ann_file = args.ann_file or cfg.data.train.ann_file
    cache_root = args.cache_root or distill_cfg.cache_root
    queue_length = args.queue_length or distill_cfg.teacher_num_frames
    use_map_mask = distill_cfg.get("use_map_mask", True)
    ground_height_range = distill_cfg.get("ground_height_range", None)
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_train_like_dataset(cfg, ann_file, queue_length, args.seed)
    dist_no_ground = build_distiller(cfg, ann_file, cache_root, None, device)
    dist_ground = build_distiller(cfg, ann_file, cache_root, ground_height_range, device)

    all_summary = {}
    for token in args.tokens:
        example, dataset_index = load_case_example(dataset, token)
        outputs = compute_case_outputs(
            example=example,
            dist_no_ground=dist_no_ground,
            dist_ground=dist_ground,
            use_map_mask=use_map_mask,
            device=device,
        )
        outputs["summary"]["dataset_index"] = dataset_index
        outputs["summary"]["ann_file"] = ann_file
        outputs["summary"]["cache_root"] = cache_root
        outputs["summary"]["ground_height_range"] = ground_height_range

        fig_path = out_dir / f"{token}.png"
        json_path = out_dir / f"{token}.json"
        save_case_figure(token, outputs, fig_path)
        with open(json_path, "w") as f:
            json.dump(outputs["summary"], f, indent=2)
        all_summary[token] = outputs["summary"]
        print(f"saved {fig_path}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
