# trainer.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


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


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")

    loss_sum = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Eval", leave=False)
    for batch in pbar:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, *_ = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        logits = model(x)
        loss_sum += float(ce(logits, y).item())

        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())

        pbar.set_postfix(acc=f"{correct/max(total,1):.4f}")

    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()

    loss_meter = 0.0
    acc_meter = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, *_ = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = ce(logits, y)
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()  # 🔑 THIS WAS MISSING

        bsz = int(y.size(0))
        n_samples += bsz
        loss_meter += float(loss.item()) * bsz
        acc_meter += _accuracy_from_logits(logits.detach(), y) * bsz

        pbar.set_postfix(
            loss=f"{loss_meter/max(n_samples,1):.4f}",
            acc=f"{acc_meter/max(n_samples,1):.4f}",
        )

    return {
        "loss": loss_meter / max(n_samples, 1),
        "acc": acc_meter / max(n_samples, 1),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    cfg: TrainConfig,
) -> Tuple[nn.Module, Dict[str, float]]:

    if cfg.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    elif cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    best_state = None
    best_metrics = {}

    for epoch in range(cfg.epochs):
        t0 = time.time()

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=cfg.grad_clip,
        )

        if val_loader is not None:
            val_stats = evaluate(model, val_loader, device)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"epoch": epoch, **val_stats}
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"epoch": epoch, **train_stats}

        dt = time.time() - t0
        if val_loader is not None:
            print(
                f"[E{epoch+1:03d}/{cfg.epochs:03d}] "
                f"train loss={train_stats['loss']:.4f} acc={train_stats['acc']:.4f} | "
                f"val loss={val_stats['loss']:.4f} acc={val_stats['acc']:.4f} | "
                f"{dt:.1f}s"
            )
        else:
            print(
                f"[E{epoch+1:03d}/{cfg.epochs:03d}] "
                f"train loss={train_stats['loss']:.4f} acc={train_stats['acc']:.4f} | "
                f"{dt:.1f}s"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics


