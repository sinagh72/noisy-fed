import numpy as np
import torch
import torch.nn.functional as F


def rand_bbox(W: int, H: int, lam: float, rng: np.random.RandomState):
    """
    Returns bbox coordinates for CutMix.
    """
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = rng.randint(W)
    cy = rng.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def apply_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float = 1.0,
    p: float = 0.5,
    rng: np.random.RandomState | None = None,
):
    """
    x: [B,C,H,W]
    y: [B] (int class)
    Returns:
      x_cutmix, y_a, y_b, lam
    """
    if rng is None:
        rng = np.random.RandomState()

    if alpha <= 0 or (rng.rand() > p):
        return x, y, y, 1.0

    B, C, H, W = x.shape
    perm = torch.randperm(B, device=x.device)

    y_a = y
    y_b = y[perm]

    lam = rng.beta(alpha, alpha)
    x1, y1, x2, y2 = rand_bbox(W, H, lam, rng)

    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]

    # adjust lambda based on exact area used
    area = (x2 - x1) * (y2 - y1)
    lam = 1.0 - (area / float(W * H))

    return x_cut, y_a, y_b, float(lam)


def cutmix_loss(
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    base_criterion,
):
    """
    base_criterion: e.g. nn.CrossEntropyLoss(reduction="mean")
    """
    return lam * base_criterion(logits, y_a) + (1.0 - lam) * base_criterion(logits, y_b)