import torch
import torch.nn as nn
import torch.nn.functional as F
from relabel import choose_k_by_ratio

class SupConLoss(nn.Module):
    """
    Supervised contrastive loss on a single view.
    Works with features: [B, D], labels: [B]
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(feats, dim=1)  # cosine space
        B = feats.size(0)

        # similarity matrix [B,B]
        sim = (feats @ feats.t()) / self.tau

        # mask out self-contrast
        self_mask = torch.eye(B, device=feats.device, dtype=torch.bool)
        sim = sim.masked_fill(self_mask, -1e9)

        # positives: same label, excluding self
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & (~self_mask)

        # log-softmax over rows
        log_prob = F.log_softmax(sim, dim=1)

        # mean over positives per anchor (skip anchors with no positives)
        pos_count = pos_mask.sum(dim=1)  # [B]
        loss_per = -(log_prob * pos_mask.float()).sum(dim=1) / (pos_count.clamp_min(1).float())

        valid = pos_count > 0
        if valid.any():
            return loss_per[valid].mean()
        # if batch has no positive pairs (rare if batchsize small), return 0
        return feats.new_tensor(0.0)


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
    num_classes: int,
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
