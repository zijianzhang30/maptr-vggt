"""Offline gradient-cosine analysis for MapTR VGGT distillation.

This module is intentionally a small reusable core: the training/evaluation
entrypoint should provide the two scalar losses and the shared parameters.
It avoids changing the training job and is suitable for checkpoint analysis.
"""
import torch
import argparse, os, json
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmcv.parallel import scatter


def flatten_grads(loss, params, retain_graph=True):
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph,
        allow_unused=True, create_graph=False)
    out = []
    for p, g in zip(params, grads):
        out.append(torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1))
    return torch.cat(out)


def gradient_cosine(loss_map, loss_dist, shared_params):
    """Return rho and both gradient norms for one fixed training batch."""
    params = [p for p in shared_params if p.requires_grad]
    g_map = flatten_grads(loss_map, params, retain_graph=True)
    g_dist = flatten_grads(loss_dist, params, retain_graph=False)
    n_map, n_dist = torch.linalg.vector_norm(g_map), torch.linalg.vector_norm(g_dist)
    rho = torch.dot(g_map, g_dist) / (n_map * n_dist).clamp_min(1e-12)
    return {"rho": float(rho.cpu()), "map_grad_norm": float(n_map.cpu()),
            "dist_grad_norm": float(n_dist.cpu())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--batches', type=int, default=4)
    ap.add_argument('--out', default='gradient_cosine.json')
    args = ap.parse_args()
    cfg = Config.fromfile(args.config)
    # Register MapTRv2 custom datasets/modules before building the dataset.
    import projects.mmdet3d_plugin  # noqa: F401
    cfg.model.pretrained = None
    cfg.data.workers_per_gpu = 0
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()
    # All parameters in the image/BEV feature path are shared by map and distillation losses.
    shared = [p for n, p in model.named_parameters()
              if p.requires_grad and ('img_backbone' in n or 'img_neck' in n or 'pts_bbox_head' in n)]
    results = {}
    for ckpt in args.checkpoints:
        load_checkpoint(model, ckpt, map_location='cuda')
        vals = []
        valid = 0
        for i, data in enumerate(loader):
            if valid >= args.batches: break
            print('[batch]', i, 'valid', valid, flush=True)
            # Use MMCV's official scatter to preserve DataContainer semantics.
            data = scatter(data, [torch.cuda.current_device()])[0]
            try:
                losses = model(return_loss=True, **data)
            except RuntimeError as exc:
                # Some nuScenes samples can have empty one-to-many targets;
                # the legacy focal-loss CUDA op cannot reduce an empty tensor.
                if 'input.numel() == 0' in str(exc) or 'empty' in str(exc):
                    print('[skip empty-target batch]', i, flush=True)
                    continue
                raise
            dist = sum(v for k, v in losses.items() if 'vggt' in k)
            main = sum(v for k, v in losses.items() if 'vggt' not in k)
            vals.append(gradient_cosine(main, dist, shared))
            valid += 1
        results[os.path.basename(ckpt)] = vals
    with open(args.out, 'w') as f: json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
