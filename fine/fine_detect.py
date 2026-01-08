import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


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
        for x, y, idx, ygt in pbar:
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



def fine_detect_clean_indices(
    feats, y_noisy, num_classes, zeta=0.5, eps=1e-12
):
    """
    Implements Algorithm 1 idea (class-wise top eigenvector alignment + GMM).
    Inputs:
      feats:   [N, D] float numpy
      y_noisy: [N] int numpy
    Returns:
      clean_mask: [N] bool
      scores:     [N] float (alignment scores)
    """
    N, D = feats.shape
    feats_t = torch.from_numpy(feats).float()  # [N, D]

    # normalize features for stable inner products (recommended)
    feats_t = feats_t / (feats_t.norm(dim=1, keepdim=True) + eps)

    # ---- compute top eigenvector per class: u_k from Sigma_k = sum z z^T ----
    u = [None] * num_classes
    for k in range(num_classes):
        idx_k = np.where(y_noisy == k)[0]
        if idx_k.size < 2:
            continue
        Zk = feats_t[idx_k]                    # [nk, D]
        Sigma = Zk.t().mm(Zk)                  # [D, D]
        # top eigenvector
        evals, evecs = torch.linalg.eigh(Sigma)  # ascending
        u_k = evecs[:, -1]                      # [D]
        u[k] = u_k / (u_k.norm() + eps)

    # ---- compute alignment scores fi = <u_{y_i}, z_i>^2 ----
    scores = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        k = int(y_noisy[i])
        if u[k] is None:
            scores[i] = 0.0
        else:
            zi = feats_t[i]
            scores[i] = float(torch.dot(u[k], zi).abs().pow(2).item())

    # ---- class-wise 2-component GMM: pick component with larger mean as clean ----
    clean_mask = np.zeros((N,), dtype=bool)
    for k in range(num_classes):
        idx_k = np.where(y_noisy == k)[0]
        if idx_k.size < 10:
            # too small -> keep all (or none); here keep all to avoid collapse
            clean_mask[idx_k] = True
            continue

        sk = scores[idx_k].reshape(-1, 1).astype(np.float32)
        gmm = GaussianMixture(n_components=2, covariance_type="full", max_iter=200, tol=1e-6)
        gmm.fit(sk)
        prob = gmm.predict_proba(sk)  # [nk,2]

        clean_comp = int(np.argmax(gmm.means_.reshape(-1)))  # higher mean => clean
        clean_prob = prob[:, clean_comp]

        clean_mask[idx_k] = (clean_prob >= zeta)

    return clean_mask, scores


def detection_metrics_from_mask(clean_mask, y_noisy, y_clean):
    """
    Treat "clean" as positive class (like the paper).
    True clean = (y_noisy == y_clean)
    """
    true_clean = (y_noisy == y_clean)

    tp = np.sum(clean_mask & true_clean)
    fp = np.sum(clean_mask & (~true_clean))
    fn = np.sum((~clean_mask) & true_clean)
    tn = np.sum((~clean_mask) & (~true_clean))

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    acc       = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=float(precision), recall=float(recall),
                f1=float(f1), accuracy=float(acc))



def noise_id_metrics(clean_mask: np.ndarray, y_noisy: np.ndarray, y_clean: np.ndarray):
    """
    clean_mask: True means predicted clean, False means predicted noisy
    """
    true_noisy = (y_noisy != y_clean)
    pred_noisy = (~clean_mask)

    tp = np.sum(pred_noisy & true_noisy)
    fp = np.sum(pred_noisy & (~true_noisy))
    fn = np.sum((~pred_noisy) & true_noisy)
    tn = np.sum((~pred_noisy) & (~true_noisy))

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    acc       = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }



def per_class_noise_stats(clean_mask: np.ndarray, y_noisy: np.ndarray, y_clean: np.ndarray, num_classes: int):
    true_noisy = (y_noisy != y_clean)
    pred_noisy = (~clean_mask)

    rows = []
    for k in range(num_classes):
        m = (y_noisy == k)
        Nk = int(m.sum())
        if Nk == 0:
            rows.append({
                "class": k, "N": 0,
                "TP": 0, "FP": 0, "FN": 0, "TN": 0,
                "TP_rate": 0.0, "FP_rate": 0.0, "FN_rate": 0.0, "TN_rate": 0.0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
            })
            continue

        TP = int(np.sum(m & pred_noisy & true_noisy))
        FP = int(np.sum(m & pred_noisy & (~true_noisy)))
        FN = int(np.sum(m & (~pred_noisy) & true_noisy))
        TN = int(np.sum(m & (~pred_noisy) & (~true_noisy)))

        # rates over all samples of that class (by noisy label)
        TP_rate = TP / Nk
        FP_rate = FP / Nk
        FN_rate = FN / Nk
        TN_rate = TN / Nk

        prec = TP / (TP + FP + 1e-12)
        rec  = TP / (TP + FN + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)

        rows.append({
            "class": k, "N": Nk,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "TP_rate": TP_rate, "FP_rate": FP_rate, "FN_rate": FN_rate, "TN_rate": TN_rate,
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
        })

    return rows
