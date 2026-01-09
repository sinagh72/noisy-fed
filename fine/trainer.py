# trainer.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

LabelMode = Literal["noisy", "clean"]  # which label to use from DualLabelDataset

@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "sgd"   # "sgd" or "adamw"
    grad_clip: float = 0.0
    log_every: int = 50
    num_classes: Optional[int] = None


def _select_y(batch, label_mode: str):
    """
    batch can be:
      (x, y)  OR  (x, y_noisy, y_clean, idx)
    """
    if len(batch) == 2:
        x, y = batch
        return x, y
    if len(batch) >= 4:
        x, y_noisy, y_clean, idx = batch[:4]
        y = y_noisy if label_mode == "noisy" else y_clean
        return x, y
    raise ValueError(f"Unexpected batch format len={len(batch)}")



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
    label_mode: str = "noisy",   # <-- NEW
) -> Dict[str, float]:
    assert label_mode in ["noisy", "clean"]
    model.train()
    ce = nn.CrossEntropyLoss()

    loss_meter = 0.0
    acc_meter = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc=f"Train({label_mode})", leave=False)
    for batch in pbar:
        x, y = _select_y(batch, label_mode)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bsz = int(y.size(0))
        n_samples += bsz
        loss_meter += float(loss.item()) * bsz
        acc_meter += (logits.argmax(dim=1) == y).float().mean().item() * bsz

        pbar.set_postfix(
            loss=f"{loss_meter/max(n_samples,1):.4f}",
            acc=f"{acc_meter/max(n_samples,1):.4f}",
        )

    return {"loss": loss_meter / max(n_samples, 1),
            "acc":  acc_meter / max(n_samples, 1)}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    cfg: TrainConfig,
    label_mode: str = "noisy",  
) -> Tuple[nn.Module, Dict[str, float]]:
    assert label_mode in ["noisy", "clean"]

    if cfg.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay, nesterov=True,
        )
    elif cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    best_state = None
    best_metrics = {}

    for epoch in range(cfg.epochs):
        t0 = time.time()

        train_stats = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device,
            grad_clip=cfg.grad_clip, label_mode=label_mode
        )

        if val_loader is not None:
            val_stats = evaluate(model, val_loader, device, label_mode="clean")  # val/test always clean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"epoch": epoch, **val_stats}
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"epoch": epoch, **train_stats}

        dt = time.time() - t0
        print(
            f"[E{epoch+1:03d}/{cfg.epochs:03d}] "
            f"train({label_mode}) loss={train_stats['loss']:.4f} acc={train_stats['acc']:.4f} | "
            f"{dt:.1f}s"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def _safe_div(a: int, b: int) -> float:
    return float(a) / float(b) if b > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int | None = None, average: str = "macro", label_mode: str = "clean"):
    """
    If loader yields dual labels, pick label_mode ("clean" for test).
    """
    assert label_mode in ["noisy", "clean"]
    model.eval()

    ce_sum = 0.0
    total = 0
    logits_all = []
    y_all = []

    for batch in loader:
        x, y = _select_y(batch, label_mode)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        logits = model(x)
        ce_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
        total += int(y.numel())

        logits_all.append(logits.detach().cpu())
        y_all.append(y.detach().cpu())

    logits_all = torch.cat(logits_all, dim=0) if logits_all else torch.empty((0, 0))
    y_all = torch.cat(y_all, dim=0).long() if y_all else torch.empty((0,), dtype=torch.long)

    if logits_all.numel() == 0:
        return {"loss": 0.0, "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auroc": float("nan"), "auprc": float("nan")}

    if num_classes is None:
        num_classes = int(logits_all.shape[1])

    preds = logits_all.argmax(dim=1)
    acc = (preds == y_all).float().mean().item()

    precision_k, recall_k, f1_k = [], [], []
    for k in range(num_classes):
        yk = (y_all == k)
        pk = (preds == k)
        tp = int((pk & yk).sum().item())
        fp = int((pk & (~yk)).sum().item())
        fn = int(((~pk) & yk).sum().item())

        prec = _safe_div(tp, tp + fp)
        rec  = _safe_div(tp, tp + fn)
        f1   = _safe_div(2 * prec * rec, prec + rec)

        precision_k.append(float(prec))
        recall_k.append(float(rec))
        f1_k.append(float(f1))

    precision_macro = float(np.mean(precision_k)) if precision_k else 0.0
    recall_macro    = float(np.mean(recall_k)) if recall_k else 0.0
    f1_macro        = float(np.mean(f1_k)) if f1_k else 0.0

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        probs = torch.softmax(logits_all, dim=1).numpy()
        y_np = y_all.numpy()

        if num_classes == 2:
            auroc = float(roc_auc_score(y_np, probs[:, 1]))
            auprc = float(average_precision_score(y_np, probs[:, 1]))
        else:
            y_oh = np.eye(num_classes, dtype=np.int32)[y_np]
            auroc = float(roc_auc_score(y_oh, probs, average=average, multi_class="ovr"))
            auprc = float(average_precision_score(y_oh, probs, average=average))
    except Exception:
        auroc = float("nan")
        auprc = float("nan")

    return {
        "loss": ce_sum / max(total, 1),
        "acc": acc,
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        "auroc": auroc,
        "auprc": auprc,
    }


@torch.no_grad()
def predict_labels_for_indices(model, dataloader, indices, device, return_probs=False):
    """
    Predict labels only for samples whose global idx is in `indices`.
    Assumes dataloader yields: (x, y_noisy, y_clean, idx_global)
    """
    model.eval()
    want = set(map(int, np.asarray(indices).tolist()))

    pred_map, conf_map = {}, {}
    prob_map = {} if return_probs else None

    for batch in tqdm(dataloader, desc="Relabel noisy samples", leave=False):
        x, _, _, idx = batch[:4]
        idx_np = idx.detach().cpu().numpy().astype(int)

        keep = np.array([g in want for g in idx_np], dtype=bool)
        if not np.any(keep):
            continue

        x = x[keep].to(device, non_blocking=True)
        idx_keep = idx_np[keep]

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        conf, pred = torch.max(probs, dim=1)
        pred = pred.detach().cpu().numpy()
        conf = conf.detach().cpu().numpy()

        if return_probs:
            probs_np = probs.detach().cpu().numpy()

        for i, g in enumerate(idx_keep):
            pred_map[int(g)] = int(pred[i])
            conf_map[int(g)] = float(conf[i])
            if return_probs:
                prob_map[int(g)] = probs_np[i].astype(np.float32)

    if return_probs:
        return pred_map, conf_map, prob_map
    return pred_map, conf_map


@torch.no_grad()
def extract_backbone_features(model, loader, device):
    """
    Returns:
      feats:  [N, D] float32 numpy
      y_noisy:[N] int64 numpy
      idx:    [N] int64 numpy (global indices)
      y_clean:[N] int64 numpy
    Assumes loader yields: x, y_noisy, idx, y_clean
    """
    model.eval()
    all_f, all_y, all_idx, all_gt = [], [], [], []

    with tqdm(loader, desc="Extracting features", leave=False) as pbar:
        for x, y, ygt, idx in pbar:
            x = x.to(device, non_blocking=True)

            # backbone features
            z = model.backbone(x)  # [B, D]
            if isinstance(z, (tuple, list)):
                z = z[0]

            all_f.append(z.detach().cpu())
            all_y.append(y.detach().cpu())
            all_idx.append(idx.detach().cpu())
            all_gt.append(ygt.detach().cpu())

    feats = torch.cat(all_f, dim=0).float().numpy()
    y_noisy = torch.cat(all_y, dim=0).long().numpy()
    idx = torch.cat(all_idx, dim=0).long().numpy()
    y_clean = torch.cat(all_gt, dim=0).long().numpy()
    return feats, y_noisy, idx, y_clean