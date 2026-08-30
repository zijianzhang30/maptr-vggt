import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASSES = ["divider", "ped_crossing", "boundary"]
COLORS = {
    "divider": "#e74c3c",
    "ped_crossing": "#2e86de",
    "boundary": "#27ae60",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Mine MapTR bad cases from eval outputs")
    parser.add_argument("--pred-json", required=True, help="Path to nuscmap_results.json")
    parser.add_argument("--gt-json", required=True, help="Path to nuscenes_map_anns_val*.json")
    parser.add_argument("--infos-pkl", required=True, help="Path to nuscenes*_val.pkl")
    parser.add_argument("--out-dir", required=True, help="Directory for summaries and plots")
    parser.add_argument(
        "--conf-thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7, 0.9],
        help="Confidence thresholds for mismatch mining",
    )
    parser.add_argument("--top-k", type=int, default=12, help="How many bad cases to save per threshold")
    return parser.parse_args()


def load_inputs(pred_json, gt_json, infos_pkl):
    pred = json.load(open(pred_json))["results"]
    gt = json.load(open(gt_json))["GTs"]
    infos = pickle.load(open(infos_pkl, "rb"))["infos"]
    info_by_token = {item["token"]: item for item in infos}
    return pred, gt, info_by_token


def count_vectors(vectors, conf_thr=None):
    counter = Counter()
    for vec in vectors:
        if conf_thr is not None and vec.get("confidence_level", 1.0) < conf_thr:
            continue
        counter[vec["cls_name"]] += 1
    return counter


def build_rows(pred, gt, info_by_token, conf_thr):
    rows = []
    for pred_item, gt_item in zip(pred, gt):
        token = pred_item["sample_token"]
        assert token == gt_item["sample_token"]
        pred_counts = count_vectors(pred_item["vectors"], conf_thr)
        gt_counts = count_vectors(gt_item["vectors"])
        row = {
            "sample_token": token,
            "map_location": info_by_token[token]["map_location"],
            "timestamp": info_by_token[token]["timestamp"],
            "pred_vectors": pred_item["vectors"],
            "gt_vectors": gt_item["vectors"],
        }
        total_abs_mismatch = 0
        total_over = 0
        total_under = 0
        for cls_name in CLASSES:
            pred_num = pred_counts[cls_name]
            gt_num = gt_counts[cls_name]
            diff = pred_num - gt_num
            row[f"pred_{cls_name}"] = pred_num
            row[f"gt_{cls_name}"] = gt_num
            row[f"diff_{cls_name}"] = diff
            total_abs_mismatch += abs(diff)
            total_over += max(diff, 0)
            total_under += max(-diff, 0)
        row["total_abs_mismatch"] = total_abs_mismatch
        row["total_overpredict"] = total_over
        row["total_underpredict"] = total_under
        rows.append(row)
    return rows


def summarize_rows(rows, top_k):
    overall = sorted(rows, key=lambda x: (-x["total_abs_mismatch"], -x["total_overpredict"], -x["total_underpredict"]))[:top_k]
    overpredict = sorted(rows, key=lambda x: (-x["total_overpredict"], -x["diff_ped_crossing"], -x["diff_boundary"]))[:top_k]
    underpredict = sorted(rows, key=lambda x: (-x["total_underpredict"], x["diff_divider"], x["diff_boundary"]))[:top_k]
    return overall, overpredict, underpredict


def _plot_vectors(ax, vectors, alpha, linestyle):
    for vec in vectors:
        pts = np.asarray(vec["pts"], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        cls_name = vec["cls_name"]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            linestyle=linestyle,
            linewidth=1.8,
            color=COLORS[cls_name],
            alpha=alpha,
        )


def save_case_plot(row, out_path, conf_thr):
    fig, ax = plt.subplots(figsize=(6, 10))
    _plot_vectors(ax, row["gt_vectors"], alpha=0.95, linestyle="-")
    pred_vectors = [
        vec for vec in row["pred_vectors"]
        if vec.get("confidence_level", 1.0) >= conf_thr
    ]
    _plot_vectors(ax, pred_vectors, alpha=0.75, linestyle="--")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-30, 30)
    ax.set_aspect("equal")
    ax.set_title(
        f"{row['sample_token']}\n{row['map_location']} | thr={conf_thr} | "
        f"mismatch={row['total_abs_mismatch']} over={row['total_overpredict']} under={row['total_underpredict']}"
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def slim_row(row):
    keep = {
        "sample_token": row["sample_token"],
        "map_location": row["map_location"],
        "timestamp": row["timestamp"],
        "total_abs_mismatch": row["total_abs_mismatch"],
        "total_overpredict": row["total_overpredict"],
        "total_underpredict": row["total_underpredict"],
    }
    for cls_name in CLASSES:
        keep[f"pred_{cls_name}"] = row[f"pred_{cls_name}"]
        keep[f"gt_{cls_name}"] = row[f"gt_{cls_name}"]
        keep[f"diff_{cls_name}"] = row[f"diff_{cls_name}"]
    return keep


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred, gt, info_by_token = load_inputs(args.pred_json, args.gt_json, args.infos_pkl)

    summary = {}
    for conf_thr in args.conf_thresholds:
        rows = build_rows(pred, gt, info_by_token, conf_thr)
        overall, overpredict, underpredict = summarize_rows(rows, args.top_k)
        tag = f"thr_{str(conf_thr).replace('.', '_')}"
        summary[tag] = {
            "overall": [slim_row(row) for row in overall],
            "overpredict": [slim_row(row) for row in overpredict],
            "underpredict": [slim_row(row) for row in underpredict],
        }

        plot_dir = out_dir / tag
        plot_dir.mkdir(exist_ok=True)
        for idx, row in enumerate(overall[: min(args.top_k, 6)]):
            save_case_plot(row, plot_dir / f"{idx:02d}_{row['sample_token']}.png", conf_thr)

    with open(out_dir / "bad_case_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {out_dir / 'bad_case_summary.json'}")
    for conf_thr in args.conf_thresholds:
        tag = f"thr_{str(conf_thr).replace('.', '_')}"
        top = summary[tag]["overall"][:3]
        print(f"\nTop mismatch cases @ conf>={conf_thr}")
        for row in top:
            print(row)


if __name__ == "__main__":
    main()
