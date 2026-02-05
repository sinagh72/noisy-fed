# trainer.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from loss import SupConLoss, sr_sn_alignment_loss, eigen_value_keep_hinge_loss, proto_pull_push_loss, init_prototypes_once, update_prototypes_ema, eigen_repulsion_loss

LabelMode = Literal["noisy", "clean"]  # which label to use from DualLabelDataset


@dataclass
class LogContext:
    round: int | None = None
    client: int | None = None
    stage: str | None = None     # e.g., "S0-noisy", "S3-corr"


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


def _select_y(batch, label_mode: str, allowed_idx_set=None, y_override_map=None):
    """
    Returns:
      x, y  (both tensors on the same device as batch)
    If filtering removes the entire batch, returns (None, None).
    """
    if len(batch) == 2:
        x, y = batch
        return x, y

    x, y_noisy, y_clean, idx = batch[:4]
    y = y_noisy if label_mode == "noisy" else y_clean

    # no filtering / overriding
    if allowed_idx_set is None and y_override_map is None:
        return x, y

    idx_np = idx.detach().cpu().numpy().astype(int)

    # filter by allowed indices
    if allowed_idx_set is not None:
        keep = np.array([i in allowed_idx_set for i in idx_np], dtype=bool)
    else:
        keep = np.ones_like(idx_np, dtype=bool)

    if not np.any(keep):
        return None, None

    x = x[keep]
    y = y[keep]
    idx_keep = idx_np[keep]

    # override labels
    if y_override_map is not None:
        # IMPORTANT: only override indices that exist in the map
        # (should be true if you built allowed_idx_set/y_override_map consistently)
        y_new = np.array([y_override_map[int(i)] for i in idx_keep], dtype=np.int64)
        y = torch.from_numpy(y_new).to(y.device)

    return x, y



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
    label_mode: str = "noisy", 
    ctx: Optional[LogContext] = None, 
    w_eig: float = 0.01,
    num_classes: int = 10,
    eig_power_iters: int = 5,
    w_keep: float = 0.001,
    allowed_idx_set=None,
    y_override_map=None,
) -> Dict[str, float]:
    assert label_mode in ["noisy", "clean"]
    model.train()

    ce = nn.CrossEntropyLoss()

    loss_meter = 0.0
    acc_meter = 0.0
    n_samples = 0

    prefix = []
    if ctx is not None:
        if ctx.round is not None:  prefix.append(f"R{ctx.round:02d}")
        if ctx.client is not None: prefix.append(f"C{ctx.client:03d}")
        if ctx.stage:              prefix.append(str(ctx.stage))

    pdesc = " ".join(prefix) + (": " if prefix else "")
    pbar = tqdm(loader, desc=f"{pdesc}Train({label_mode})", leave=False)

    for batch in pbar:
        x, y = _select_y(batch, label_mode, allowed_idx_set, y_override_map)
        if x is None:
            continue

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        logits, feats = model(x, return_feats=True)
        loss_ce = ce(logits, y)



        # loss = loss_ce + w_eig * loss_eig + w_keep * loss_keep
        # loss_align = sr_sn_alignment_loss(
        #     feats=feats,              # use raw feats; the function normalizes internally
        #     y=y,
        #     num_classes=num_classes,
        #     k_noise=4,                # tune
        #     w_sr=1.0,
        #     w_sn=0.5,                 # tune
        #     w_rep=0.25,               # tune
        #     sim_thr=0.2,
        #     min_count=4,
        # )

        # loss = loss_ce + 0.1 * loss_align 
        loss = loss_ce

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
    ctx: Optional[LogContext] = None,
    num_classes: int = 10,
    allowed_idx_set=None,
    y_override_map=None
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

    model.train()

    best_state = None
    best_metrics = {}

    for epoch in range(cfg.epochs):
        t0 = time.time()

        train_stats = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device,
            grad_clip=cfg.grad_clip, label_mode=label_mode, ctx=ctx, allowed_idx_set=allowed_idx_set, y_override_map=y_override_map 
        )

        if val_loader is not None:
            val_stats = evaluate(model, val_loader, device, label_mode="clean")  # val/test always clean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"epoch": epoch, **val_stats}
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"epoch": epoch, **train_stats}

        dt = time.time() - t0
        prefix = ""
        if ctx is not None:
            parts = []
            if ctx.round is not None:  parts.append(f"R{ctx.round:02d}")
            if ctx.client is not None: parts.append(f"C{ctx.client:03d}")
            if ctx.stage:              parts.append(str(ctx.stage))
            if parts:
                prefix = "[" + " ".join(parts) + "] "

        print(
            f"{prefix}"
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
    model.eval()
    by_class = {}

    for x, y_noisy, y_clean, idx in loader:
        x = x.to(device, non_blocking=True)

        z = model.extract_features(x)
        z = z.detach().cpu().float()

        y_noisy = y_noisy.detach().cpu().long()
        y_clean = y_clean.detach().cpu().long()
        idx = idx.detach().cpu().long()

        for i in range(z.size(0)):
            c = int(y_noisy[i].item())
            if c not in by_class:
                by_class[c] = {"feats": [], "idx": [], "y_noisy": [], "y_clean": []}
            by_class[c]["feats"].append(z[i:i+1])
            by_class[c]["idx"].append(idx[i:i+1])
            by_class[c]["y_noisy"].append(y_noisy[i:i+1])
            by_class[c]["y_clean"].append(y_clean[i:i+1])

    for c, rec in by_class.items():
        by_class[c] = {
            "feats":   torch.cat(rec["feats"], 0).numpy(),
            "idx":     torch.cat(rec["idx"], 0).numpy(),
            "y_noisy": torch.cat(rec["y_noisy"], 0).numpy(),
            "y_clean": torch.cat(rec["y_clean"], 0).numpy(),
        }
    return by_class




def save_model(path, model, cfg_train=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "cfg_train": vars(cfg_train) if cfg_train is not None else None,
        "extra": extra,
    }
    torch.save(payload, path)
    print(f"[OK] Saved clean model to: {path}")

def load_model(path, model, device):
    ckpt = torch.load(path, map_location=device)
    # support both "full payload" and raw state_dict
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    print(f"[OK] Loaded clean model from: {path}")
    return model, ckpt



@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y = batch[0], batch[2]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        logits = model(x)
        pred = logits.argmax(dim=1)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)
