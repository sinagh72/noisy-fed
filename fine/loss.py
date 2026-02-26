import torch
import torch.nn as nn
import torch.nn.functional as F
from relabel import choose_k_by_ratio
from dataclasses import dataclass
from typing import Literal
import math


@torch.no_grad()
def _class_pca_basis(Xc: torch.Tensor, eps: float = 1e-6):
    X = Xc - Xc.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    k = choose_k_by_ratio(S.detach().cpu().numpy(), c=10)

    v_r = Vh[:k]  # [D]

    v_n = Vh[k:]
    v_r = F.normalize(v_r, dim=1, eps=eps)
    v_n = F.normalize(v_n, dim=1, eps=eps) 
    return v_r, v_n


def sr_sn_alignment_loss(
    feats: torch.Tensor,
    y: torch.Tensor,
    w_sr: float = 1.0,
    w_sn: float = 1.0,
    w_rep: float = 0.25,
    sim_thr: float = 0.2,
    min_count: int = 4,
    eps: float = 1e-8,
):
    feats_n = F.normalize(feats, dim=1, eps=eps)

    present = []
    vR, vN = {}, {}

    for c_t in torch.unique(y):
        c = int(c_t.item())
        mask = (y == c)
        if mask.sum().item() < min_count:
            continue
        Vr, Vn = _class_pca_basis(feats_n[mask], eps=eps)
        present.append(c)
        vR[c], vN[c] = Vr, Vn

    if len(present) == 0:
        return feats.new_tensor(0.0)

    # --- SR + SN ---
    sr_losses, sn_losses = [], []
    for c in present:
        fc = feats_n[y == c]  # [n_c, D]

        Vr = vR[c]            # [k_c, D]
        proj_r = fc @ Vr.t()                        # [n_c, k_c]
        sr_energy = (proj_r ** 2).sum(dim=1)        # [n_c] in [0,1]
        sr_losses.append((1.0 - sr_energy).mean())

        Vn = vN[c]
        if Vn.numel() > 0:
            proj_n = fc @ Vn.t()
            sn_losses.append((proj_n ** 2).sum(dim=1).mean())
        else:
            sn_losses.append(fc.new_tensor(0.0))

    loss_sr = torch.stack(sr_losses).mean()
    loss_sn = torch.stack(sn_losses).mean()

    # --- subspace-overlap repulsion ---
    loss_rep = feats.new_tensor(0.0)
    if w_rep > 0 and len(present) >= 2:
        overlap = {}
        for i in present:
            for j in present:
                if i == j:
                    continue
                Vi, Vj = vR[i], vR[j]     # [k_i,D], [k_j,D]
                kij = min(Vi.size(0), Vj.size(0))
                A = Vi @ Vj.t()           # [k_i,k_j]
                sij = (A.pow(2).sum() / (kij + eps))  # in [0,1]
                overlap[(i, j)] = sij

        rep_terms = []
        for i in present:
            fi = feats_n[y == i]          # [n_i,D]
            if fi.numel() == 0:
                continue

            weights, energies = [], []
            for j in present:
                if j == i:
                    continue
                sij = overlap[(i, j)]
                wij = F.relu(sij - sim_thr)
                if float(wij.item()) <= 0.0:
                    continue

                Vj = vR[j]                           # [k_j,D]
                proj_ij = fi @ Vj.t()                # [n_i,k_j]
                e_ij = (proj_ij ** 2).sum(dim=1).mean()
                weights.append(wij)
                energies.append(e_ij)

            if len(weights) == 0:
                rep_terms.append(fi.new_tensor(0.0))
            else:
                W = torch.stack(weights)
                E = torch.stack(energies)
                rep_terms.append((W * E).sum() / (W.sum() + eps))

        loss_rep = torch.stack(rep_terms).mean() if len(rep_terms) else feats.new_tensor(0.0)

    return w_sr * loss_sr + w_sn * loss_sn + w_rep * loss_rep


LossMode = Literal["ce", "jal_ce", "kd", "kd_jal_ce"]

@dataclass
class LossConfig:
    mode: LossMode = "ce"

    # ---- JAL (paper: JAL-CE = α·NCE + β·AMSE) ----
    jal_alpha: float = 1.0       # α
    jal_beta: float = 1.0        # β
    amse_a: float = 20.0         # "a" in AMSE (>=1). Paper uses 10~40 depending on noise. :contentReference[oaicite:1]{index=1}

    # ---- KD ----
    kd_temp: float = 4.0         # T
    kd_lambda: float = 0.5       # weight on KD term
    kd_hard_weight: float = 1.0  # weight on hard-label term (CE/JAL)


class AMSELoss(nn.Module):
    def __init__(self, a: float = 1.0):
        super().__init__()
        self.a = float(a)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # AMSE in paper: (1/K) || a·e_y - p ||^2   with p = softmax(logits) :contentReference[oaicite:2]{index=2}
        p = F.softmax(logits, dim=1)
        y1h = F.one_hot(target, num_classes=p.size(1)).float().to(p.device)
        y1h = y1h * self.a
        return ((p - y1h) ** 2).mean()


class NCELoss(nn.Module):
    """
    NCE = CE / sum_k CE(pred, k) (normalized CE), from APL-style framework. :contentReference[oaicite:3]{index=3}
    Implementation: for each sample, denominator is sum over all possible labels.
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # CE for true label
        ce_true = F.cross_entropy(logits, target, reduction="none")  # [B]

        # denominator: sum_k CE(logits, k)
        # CE(logits,k) = -log softmax_k
        logp = F.log_softmax(logits, dim=1)                          # [B,K]
        ce_all = -logp                                               # [B,K]
        denom = ce_all.sum(dim=1).clamp_min(self.eps)                # [B]

        nce = ce_true / denom
        return nce.mean()


class NormalizedFocalLoss(nn.Module):
    """
    Paper's NormalizedFocalLoss:
      loss = scale * FL / normalizor
    where
      FL = -(1-pt)^gamma * logpt
      normalizor = sum_k [-(1-pk)^gamma * logpk]   (per sample)
    """
    def __init__(self, gamma: float = 0.5, scale: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.gamma = float(gamma)
        self.scale = float(scale)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target: [B]
        target = target.view(-1, 1)

        logp = F.log_softmax(logits, dim=1)                 # [B,K]
        p = logp.exp()

        # normalizor = sum_k [-(1-pk)^gamma * logpk]
        normalizor = torch.sum(-1.0 * (1.0 - p).pow(self.gamma) * logp, dim=1)  # [B]
        normalizor = normalizor.clamp_min(self.eps)

        logpt = logp.gather(1, target).view(-1)             # [B]
        pt = logpt.exp()

        fl = -1.0 * (1.0 - pt).pow(self.gamma) * logpt      # [B]
        loss = self.scale * fl / normalizor                 # [B]
        return loss.mean()
    

class JAL_CE_Loss(nn.Module):
    """
    JAL-CE = α·NCE + β·AMSE :contentReference[oaicite:4]{index=4}
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, amse_a: float = 20.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.nce = NCELoss()
        self.amse = AMSELoss(a=amse_a)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.nce(logits, target) + self.beta * self.amse(logits, target)


class JAL_Focal_Loss(nn.Module):
    """
    JAL-Focal (paper: NFLandAMSE):
      α·NormalizedFocalLoss + β·AMSE
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        amse_a: float = 20.0,
        gamma: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.nfl = NormalizedFocalLoss(gamma=gamma, scale=1.0, eps=eps)
        self.amse = AMSELoss(a=amse_a)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.nfl(logits, target) + self.beta * self.amse(logits, target)



def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    """
    Standard KD: KL( soft_t || soft_s ) * T^2
    """
    T = float(T)
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


def build_loss_fn(loss_cfg: LossConfig) -> nn.Module:
    if loss_cfg.mode == "ce":
        return nn.CrossEntropyLoss()

    if loss_cfg.mode == "jal_ce":
        return JAL_CE_Loss(alpha=loss_cfg.jal_alpha, beta=loss_cfg.jal_beta, amse_a=loss_cfg.amse_a)

    if loss_cfg.mode == "jal_focal":
        return JAL_Focal_Loss(
            alpha=loss_cfg.jal_alpha,
            beta=loss_cfg.jal_beta,
            amse_a=loss_cfg.amse_a,
            gamma=loss_cfg.focal_gamma,
            eps=loss_cfg.focal_eps,
        )

    # KD modes: hard loss returned here; KD KL handled in train loop
    if loss_cfg.mode == "kd":
        return nn.CrossEntropyLoss()

    if loss_cfg.mode == "kd_jal_ce":
        return JAL_CE_Loss(alpha=loss_cfg.jal_alpha, beta=loss_cfg.jal_beta, amse_a=loss_cfg.amse_a)

    if loss_cfg.mode == "kd_jal_focal":
        return JAL_Focal_Loss(
            alpha=loss_cfg.jal_alpha,
            beta=loss_cfg.jal_beta,
            amse_a=loss_cfg.amse_a,
            gamma=loss_cfg.focal_gamma,
            eps=loss_cfg.focal_eps,
        )

    raise ValueError(f"Unknown loss mode: {loss_cfg.mode}")


def amse_a_cosine(round_idx: int,
                  total_rounds: int = 100,
                  a_start: float = 40.0,
                  a_end: float = 20.0,
                  gamma: float = 3.0) -> float:
    """
    Cosine decay with power gamma.
    gamma > 1 -> faster early drop.
    """
    r = max(1, min(round_idx, total_rounds))
    t = (r - 1) / (total_rounds - 1)  # normalize to [0,1]
    cosine_part = math.cos(math.pi / 2 * t) ** gamma
    a = a_end + (a_start - a_end) * cosine_part

    return float(a)