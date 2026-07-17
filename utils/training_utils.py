"""Small, CPU-testable helpers used by the 3DGS training loop."""

import re

import torch


def natural_image_key(image_name):
    """Sort image names by their numeric capture index when present."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", image_name)
    )


def evenly_spaced_holdout_indices(count, fraction):
    """Return centre-of-bin indices for a uniform train-image holdout."""
    if count < 2:
        raise ValueError("at least two cameras are required for a validation split")
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    holdout_count = max(1, round(count * fraction))
    if holdout_count >= count:
        raise ValueError("validation_fraction must leave at least one training camera")
    return {
        min(count - 1, int((i + 0.5) * count / holdout_count))
        for i in range(holdout_count)
    }


def foreground_weighted_l1(rendered, target, foreground_mask, foreground_weight):
    """Mean RGB L1 with an optional spatial foreground emphasis."""
    pixel_l1 = torch.abs(rendered - target).mean(dim=0, keepdim=True)
    pixel_weight = 1.0 + foreground_weight * foreground_mask
    return (pixel_l1 * pixel_weight).sum() / (pixel_weight.sum() + 1e-6)


def foreground_edge_l1(rendered, target, foreground_mask):
    """Match horizontal and vertical RGB gradients in and around the foreground."""
    error_x = torch.abs(
        (rendered[:, :, 1:] - rendered[:, :, :-1])
        - (target[:, :, 1:] - target[:, :, :-1])
    ).mean(dim=0, keepdim=True)
    error_y = torch.abs(
        (rendered[:, 1:, :] - rendered[:, :-1, :])
        - (target[:, 1:, :] - target[:, :-1, :])
    ).mean(dim=0, keepdim=True)
    # Include either side of an edge, so a mask boundary itself is supervised.
    mask_x = torch.maximum(foreground_mask[:, :, 1:], foreground_mask[:, :, :-1])
    mask_y = torch.maximum(foreground_mask[:, 1:, :], foreground_mask[:, :-1, :])
    numerator = (error_x * mask_x).sum() + (error_y * mask_y).sum()
    denominator = mask_x.sum() + mask_y.sum() + 1e-6
    return numerator / denominator


def image_edge_l1(rendered, target, edge_strength=4.0):
    """Match image gradients, with extra weight on high-contrast target edges.

    Unlike a semantic foreground mask this remains usable when a scene has no
    masks.  It gives chair perforations, thin legs, and object silhouettes more
    gradient signal while keeping a modest weight on broad background changes.
    """
    render_x = rendered[:, :, 1:] - rendered[:, :, :-1]
    render_y = rendered[:, 1:, :] - rendered[:, :-1, :]
    target_x = target[:, :, 1:] - target[:, :, :-1]
    target_y = target[:, 1:, :] - target[:, :-1, :]

    error_x = torch.abs(render_x - target_x).mean(dim=0, keepdim=True)
    error_y = torch.abs(render_y - target_y).mean(dim=0, keepdim=True)
    target_edge_x = torch.abs(target_x).mean(dim=0, keepdim=True)
    target_edge_y = torch.abs(target_y).mean(dim=0, keepdim=True)

    # Per-image normalization makes the weighting stable across bright and dark
    # frames; the clamp prevents a handful of window highlights dominating.
    scale_x = target_edge_x.mean().clamp_min(1e-6)
    scale_y = target_edge_y.mean().clamp_min(1e-6)
    weight_x = 1.0 + edge_strength * (target_edge_x / scale_x).clamp(max=5.0)
    weight_y = 1.0 + edge_strength * (target_edge_y / scale_y).clamp(max=5.0)
    return ((error_x * weight_x).sum() + (error_y * weight_y).sum()) / (weight_x.sum() + weight_y.sum() + 1e-6)
