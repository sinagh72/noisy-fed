import torch
import torch.nn as nn
import torch.nn.functional as F

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



def proto_pull_push_loss(
    feats: torch.Tensor,
    y: torch.Tensor,
    prototypes: torch.Tensor,
    margin: float = 0.2,
    use_cosine: bool = True,
) -> torch.Tensor:
    """
    Pull to own prototype + push away from closest wrong prototype by margin.
    If use_cosine=True, operates on cosine distance in normalized space.
    """
    if use_cosine:
        feats_n = F.normalize(feats, dim=1)
        prot_n  = F.normalize(prototypes, dim=1)
        sims = feats_n @ prot_n.t()             # [B, K], higher=closer
        pos = sims.gather(1, y.view(-1, 1)).squeeze(1)   # [B]

        # mask out the true class, then take max (closest wrong)
        B, K = sims.shape
        mask = torch.ones_like(sims, dtype=torch.bool)
        mask.scatter_(1, y.view(-1, 1), False)
        neg = sims.masked_fill(~mask, -1e9).max(dim=1).values

        # want pos >= neg + margin  ->  max(0, margin + neg - pos)
        push = F.relu(margin + neg - pos).mean()
        pull = (1.0 - pos).mean()               # encourages pos -> 1
        return pull + push

    else:
        # L2 version
        B = feats.size(0)
        p_pos = prototypes[y]                   # [B,D]
        pull = F.mse_loss(feats, p_pos)

        dists = torch.cdist(feats, prototypes)  # [B,K]
        d_pos = dists.gather(1, y.view(-1, 1)).squeeze(1)

        # closest wrong prototype
        mask = torch.ones_like(dists, dtype=torch.bool)
        mask.scatter_(1, y.view(-1, 1), False)
        d_neg = dists.masked_fill(~mask, 1e9).min(dim=1).values

        # want d_neg >= d_pos + margin  -> max(0, margin + d_pos - d_neg)
        push = F.relu(margin + d_pos - d_neg).mean()
        return pull + push


@torch.no_grad()
def update_prototypes_ema(
    prototypes: torch.Tensor,   # [K,D]
    feats: torch.Tensor,        # [B,D]
    y: torch.Tensor,            # [B]
    momentum: float = 0.9,
):
    K, D = prototypes.shape
    for c in y.unique():
        c = int(c.item())
        mask = (y == c)
        if mask.any():
            mu = feats[mask].mean(dim=0)
            prototypes[c].mul_(momentum).add_(mu * (1.0 - momentum))

@torch.no_grad()
def init_prototypes_once(prototypes, proto_inited, feats, y):
    for c in y.unique():
        c = int(c.item())
        if not proto_inited[c]:
            prototypes[c] = feats[y == c].mean(dim=0)
            proto_inited[c] = True




def top_eigvec_power(Xc: torch.Tensor, n_iter: int = 5, eps: float = 1e-6):
    """
    Xc: [n, D] centered features for one class
    Returns: v [D] approx top eigenvector of (Xc^T Xc)
    """
    n, D = Xc.shape
    # random init (deterministic init also OK)
    v = F.normalize(torch.randn(D, device=Xc.device, dtype=Xc.dtype), dim=0)

    for _ in range(n_iter):
        # w = (X^T X) v  but compute without forming X^T X:
        # t = X v -> [n]
        t = Xc @ v                       # [n]
        w = Xc.t() @ t                   # [D]
        v = F.normalize(w, dim=0, eps=eps)

    return v


def eigen_repulsion_loss(
    feats: torch.Tensor,    # [B, D]
    y: torch.Tensor,        # [B]
    num_classes: int,
    n_iter: int = 5,
    min_count: int = 4,     # need enough samples to estimate covariance direction
):
    """
    Penalize similarity between top eigenvectors of each class covariance.
    Returns scalar loss.
    """
    vs = []
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum().item() < min_count:
            continue
        Xc = feats[mask]                    # [n_c, D]
        Xc = Xc - Xc.mean(dim=0, keepdim=True)
        v = top_eigvec_power(Xc, n_iter=n_iter)
        vs.append(v)

    if len(vs) < 2:
        return feats.new_tensor(0.0)

    V = torch.stack(vs, dim=0)              # [K', D]
    G = V @ V.t()                           # [K', K']  (cosine-ish since v normalized)
    off = G - torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    return (off**2).mean()                  # average squared off-diagonal similarity


def eigen_value_keep_hinge_loss(
    feats: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    lam_min: float = 0.05,
    n_iter: int = 5,
    min_count: int = 4,
):
    losses = []
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum().item() < min_count:
            continue
        Xc = feats[mask]
        Xc = Xc - Xc.mean(dim=0, keepdim=True)

        v = top_eigvec_power(Xc, n_iter=n_iter)
        lam = (Xc @ v).pow(2).mean()

        # penalize only if variance is too small
        losses.append(F.relu(lam_min - lam))

    if not losses:
        return feats.new_tensor(0.0)
    return torch.stack(losses).mean()



import torch
import torch.nn.functional as F

@torch.no_grad()
def _class_pca_basis(Xc: torch.Tensor, k_noise: int = 4, eps: float = 1e-6):
    """
    Xc: [n_c, D] features for one class (NOT normalized required, we center inside)
    Returns:
      v_r: [D] top right singular vector
      Vn:  [m, D] noise basis (next components), m = min(k_noise, D-1, n_c-1)
    """
    n, D = Xc.shape
    X = Xc - Xc.mean(dim=0, keepdim=True)

    # SVD: X = U S Vh, Vh: [D, D] (or [r, D]) depending on full_matrices
    # Using full_matrices=False gives Vh: [r, D] where r=min(n,D)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)

    v_r = Vh[0]  # [D]

    # pick next components as "noise subspace"
    max_m = min(k_noise, Vh.size(0) - 1)  # can't exceed available comps minus top
    if max_m <= 0:
        Vn = X.new_zeros((0, D))
    else:
        Vn = Vh[1:1+max_m]  # [m, D]

    v_r = F.normalize(v_r, dim=0, eps=eps)
    Vn = F.normalize(Vn, dim=1, eps=eps) if Vn.numel() > 0 else Vn
    return v_r, Vn


def sr_sn_alignment_loss(
    feats: torch.Tensor,      # [B, D]
    y: torch.Tensor,          # [B]
    num_classes: int,
    k_noise: int = 4,
    w_sr: float = 1.0,
    w_sn: float = 1.0,
    w_rep: float = 0.25,      # repulsion to similar classes' v_r
    sim_thr: float = 0.2,     # only repel if class directions are similar
    min_count: int = 4,
    eps: float = 1e-6,
):
    """
    Returns: scalar loss.
    """
    feats_n = F.normalize(feats, dim=1, eps=eps)

    # build v_r and Vn for classes present in the batch
    present = []
    vR = {}
    vN = {}
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum().item() < min_count:
            continue
        v_r_c, Vn_c = _class_pca_basis(feats_n[mask], k_noise=k_noise, eps=eps)
        present.append(c)
        vR[c] = v_r_c
        vN[c] = Vn_c

    if len(present) == 0:
        return feats.new_tensor(0.0)

    # --- SR pull + SN suppress (per-sample) ---
    sr_losses = []
    sn_losses = []

    for c in present:
        mask = (y == c)
        fc = feats_n[mask]  # [n_c, D]

        # s_r = |<f, v_r>|
        sr = torch.abs(fc @ vR[c])  # [n_c]
        sr_losses.append((1.0 - sr).mean())

        # s_n = || f @ Vn^T ||_2  (we penalize squared energy)
        Vn_c = vN[c]  # [m, D]
        if Vn_c.numel() > 0:
            proj = fc @ Vn_c.t()             # [n_c, m]
            sn_losses.append((proj.pow(2).sum(dim=1)).mean())
        else:
            sn_losses.append(fc.new_tensor(0.0))

    loss_sr = torch.stack(sr_losses).mean()
    loss_sn = torch.stack(sn_losses).mean()

    # --- repel similar classes (optional, helps when classes are close) ---
    loss_rep = feats.new_tensor(0.0)
    if w_rep > 0 and len(present) >= 2:
        V = torch.stack([vR[c] for c in present], dim=0)  # [K', D]
        S = V @ V.t()                                     # [K', K']
        # weights for similar pairs (off-diagonal)
        W = torch.relu(S - sim_thr)
        W.fill_diagonal_(0.0)

        # for each class c, penalize alignment of its samples with other classes' v_r weighted by similarity
        rep_terms = []
        for i, c in enumerate(present):
            mask = (y == c)
            fc = feats_n[mask]                            # [n_c, D]
            # alignment to all present class directions
            align = torch.abs(fc @ V.t())                 # [n_c, K']
            # exclude own column i
            align_other = align.clone()
            align_other[:, i] = 0.0
            # weight by similarity W[i, :]
            rep = (align_other * W[i].unsqueeze(0)).mean()
            rep_terms.append(rep)
        loss_rep = torch.stack(rep_terms).mean()

    return w_sr * loss_sr + w_sn * loss_sn + w_rep * loss_rep
