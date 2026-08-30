"""Memory-efficient gradient conflict metrics for distillation analysis."""
import torch


def scalarize_loss(value):
    if torch.is_tensor(value):
        return value.mean()
    if isinstance(value, (list, tuple)):
        values = [scalarize_loss(item) for item in value]
        values = [item for item in values if item is not None]
        return sum(values) if values else None
    return None


def sum_selected_losses(losses, selector):
    values = []
    for name, value in losses.items():
        if selector(name):
            value = scalarize_loss(value)
            if value is not None:
                values.append(value)
    return sum(values) if values else None


def gradient_cosine(loss_a, loss_b, params, eps=1e-12):
    params = [param for param in params if param.requires_grad]
    if not params:
        return None
    grads_a = torch.autograd.grad(loss_a, params, retain_graph=True,
                                  create_graph=False, allow_unused=True)
    grads_b = torch.autograd.grad(loss_b, params, retain_graph=True,
                                  create_graph=False, allow_unused=True)
    dot = torch.zeros((), device=loss_a.device, dtype=torch.float32)
    norm_a_sq, norm_b_sq = torch.zeros_like(dot), torch.zeros_like(dot)
    num_shared = 0
    for grad_a, grad_b in zip(grads_a, grads_b):
        if grad_a is None or grad_b is None:
            continue
        grad_a, grad_b = grad_a.detach().float(), grad_b.detach().float()
        dot += (grad_a * grad_b).sum()
        norm_a_sq += grad_a.square().sum()
        norm_b_sq += grad_b.square().sum()
        num_shared += 1
    if num_shared == 0:
        return None
    norm_a, norm_b = norm_a_sq.sqrt(), norm_b_sq.sqrt()
    cosine = dot / (norm_a * norm_b).clamp_min(eps)
    ratio = norm_b / norm_a.clamp_min(eps)
    return dict(cosine=float(cosine.cpu()),
                map_grad_norm=float(norm_a.cpu()),
                vggt_grad_norm=float(norm_b.cpu()),
                norm_ratio=float(ratio.cpu()), dot=float(dot.cpu()),
                num_shared=num_shared,
                effective_conflict=float(((-cosine).clamp_min(0) * ratio).cpu()))
