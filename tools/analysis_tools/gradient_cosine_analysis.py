"""Probe MapTR/VGGT gradient conflict on fixed batches and checkpoints."""
import argparse, csv, json, os, random, sys
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import projects.mmdet3d_plugin  # noqa: F401,E402
from projects.mmdet3d_plugin.maptr.distill.grad_conflict import gradient_cosine, sum_selected_losses


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def module_groups(model):
    candidates = {
        'img_backbone_layer4': getattr(model.img_backbone, 'layer4', None),
        'img_neck': getattr(model, 'img_neck', None),
        'bev_encoder': getattr(getattr(model.pts_bbox_head, 'transformer', None), 'encoder', None),
    }
    return {name: list(module.parameters()) for name, module in candidates.items() if module is not None}


def summarize(rows):
    result, buckets = [], {}
    for row in rows:
        key = (row['checkpoint'], row['objective'], row['group'])
        buckets.setdefault(key, []).append(row)
    metrics = ('cosine', 'map_grad_norm', 'vggt_grad_norm', 'norm_ratio', 'effective_conflict')
    for (checkpoint, objective, group), values in buckets.items():
        item = dict(checkpoint=checkpoint, objective=objective, group=group,
                    num_batches=len(values),
                    conflict_rate=sum(v['cosine'] < 0 for v in values) / len(values))
        for metric in metrics:
            array = np.asarray([v[metric] for v in values], dtype=np.float64)
            item['mean_' + metric], item['std_' + metric] = float(array.mean()), float(array.std())
        result.append(item)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('legacy_checkpoints', nargs='*')
    parser.add_argument('--checkpoints', nargs='+', default=None)
    parser.add_argument('--batches', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--samples-per-gpu', type=int, default=2)
    parser.add_argument('--out-dir', default='work_dirs/grad_conflict_probe')
    parser.add_argument('--groups', nargs='+', default=['img_backbone_layer4', 'img_neck', 'bev_encoder'])
    parser.add_argument('--objectives', nargs='+', default=['loss_vggt_img_feat', 'loss_vggt_feat'])
    return parser.parse_args()


def main():
    args = parse_args(); args.checkpoints = args.checkpoints or args.legacy_checkpoints
    if not args.checkpoints:
        raise ValueError('Provide checkpoints positionally or with --checkpoints')
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None; cfg.data.workers_per_gpu = 0
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(dataset, samples_per_gpu=args.samples_per_gpu,
                              workers_per_gpu=0, dist=False, shuffle=False)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).cuda()
    # Training graph is required: MapTRv2Head computes one-to-many queries
    # only when self.training is True.  Reset RNG before each checkpoint/batch
    # below so dropout randomness remains comparable across checkpoints.
    model.train()
    print('model.training:', model.training, flush=True)
    print('pts_bbox_head.training:', model.pts_bbox_head.training, flush=True)
    groups = module_groups(model)
    unknown = set(args.groups) - set(groups)
    if unknown: raise KeyError('Unavailable parameter groups: ' + ', '.join(sorted(unknown)))
    mmcv.mkdir_or_exist(args.out_dir); rows = []
    for checkpoint in args.checkpoints:
        load_checkpoint(model, checkpoint, map_location='cuda')
        checkpoint_name = os.path.basename(checkpoint); valid = 0; set_seed(args.seed)
        for batch_index, data in enumerate(loader):
            if valid >= args.batches: break
            set_seed(args.seed + batch_index)
            data = scatter(data, [torch.cuda.current_device()])[0]
            try:
                losses = model(return_loss=True, **data)
                print('[loss keys]', sorted(losses.keys()), flush=True)
            except RuntimeError as error:
                raise
            map_loss = sum_selected_losses(losses, lambda name: 'loss' in name and not name.startswith('loss_vggt'))
            if map_loss is None: raise RuntimeError('No MapTR losses found')
            for objective in args.objectives:
                vggt_loss = sum_selected_losses(losses, lambda name, obj=objective: name == obj)
                if vggt_loss is None: continue
                for group_name in args.groups:
                    stats = gradient_cosine(map_loss, vggt_loss, groups[group_name])
                    if stats is not None:
                        rows.append(dict(checkpoint=checkpoint_name, batch=batch_index,
                                         objective=objective, group=group_name, **stats))
            valid += 1
            print('[probe] checkpoint={} batch={}/{}'.format(checkpoint_name, valid, args.batches), flush=True)
            del losses, map_loss; torch.cuda.empty_cache()
    summary = summarize(rows)
    with open(os.path.join(args.out_dir, 'raw.json'), 'w') as handle: json.dump(rows, handle, indent=2)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as handle: json.dump(summary, handle, indent=2)
    if summary:
        with open(os.path.join(args.out_dir, 'summary.csv'), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
