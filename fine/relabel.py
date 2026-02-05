import numpy as np

def choose_k_by_ratio(S: np.ndarray, c: float = 10.0) -> int:
    """
    S: 1D array of singular values sorted descending (as returned by np.linalg.svd).
    Returns smallest k >= 1 satisfying sum_{i<=k} S_i >= c * sum_{i>k} S_i.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        return 0

    total = S.sum()
    if total <= 0:
        return 0

    target = (c / (c + 1.0)) * total  # same inequality rearranged
    prefix = np.cumsum(S)

    # first index where prefix >= target
    k0 = int(np.searchsorted(prefix, target, side="left"))  # 0-based
    k = max(1, min(S.size, k0 + 1))  # convert to 1-based k and clamp
    return k



def relabel_noisy_samples(
    feats_noisy: dict,
    clean_eig: dict,
    num_classes: int,
    mode: str = "agree_then_margin",   # "sr", "sn", "agree", "agree_then_sr", "agree_then_sn", "agree_then_margin"
    margin_thr=0.25,                   # scalar or per-class array length C (used in agree_then_margin)
    fill_with_original: bool = True,   # if still -1, replace by y_noisy
    eps: float = 1e-12,
):
    # ---- flatten samples ----
    idx_list, z_list, yno_list, ycl_list = [], [], [], []
    for c in sorted(feats_noisy.keys()):
        rec = feats_noisy[c]
        idx_list.append(rec["idx"].astype(np.int64).reshape(-1))
        z_list.append(rec["feats"].astype(np.float32))
        yno_list.append(rec["y_noisy"].astype(np.int64).reshape(-1))
        ycl_list.append(rec["y_clean"].astype(np.int64).reshape(-1))

    missing = [j for j in range(num_classes) if j not in clean_eig]
    if missing:
        raise ValueError(f"clean_eig missing classes: {missing}")

    idx_all = np.concatenate(idx_list, 0)
    Z_all   = np.concatenate(z_list, 0)          # [N,D]
    y_noisy_all = np.concatenate(yno_list, 0)
    y_clean_all = np.concatenate(ycl_list, 0)

    N, D = Z_all.shape

    # ---- normalize z for s_r cosine ----
    Z_hat = Z_all / (np.linalg.norm(Z_all, axis=1, keepdims=True) + eps)

    # ---- pack eig ----
    vR = np.zeros((num_classes, D), dtype=np.float32)
    vR_norm = np.zeros((num_classes,), dtype=np.float32)
    vN_list = []
    for j in range(num_classes):
        v_r_j, v_n_j = clean_eig[j]
        v_r_j = np.asarray(v_r_j, dtype=np.float32).reshape(-1)
        vR[j] = v_r_j
        vR_norm[j] = np.linalg.norm(v_r_j) + eps

        v_n_j = np.asarray(v_n_j, dtype=np.float32)
        if v_n_j.ndim == 1:
            v_n_j = v_n_j[None, :]
        vN_list.append(v_n_j)

    # ---- compute SR ----
    vR_hat = vR / vR_norm[:, None]
    SR = np.abs(Z_hat @ vR_hat.T)  # [N,C]

    arg_sr = SR.argmax(axis=1)              # [N]
    sr_max = SR[np.arange(N), arg_sr]       # [N]

    SR_sorted = np.sort(SR, axis=1)
    sr_second = SR_sorted[:, -2] if num_classes >= 2 else np.zeros_like(sr_max)
    sr_margin = sr_max - sr_second          # [N]

    # ---- compute SN ----
    SN = np.zeros((N, num_classes), dtype=np.float32)
    for j in range(num_classes):
        Vn = vN_list[j]
        proj = Z_all @ Vn.T                 # [N, M_j]
        SN[:, j] = np.linalg.norm(proj, axis=1)

    arg_sn = SN.argmin(axis=1)              # [N]
    sn_min = SN[np.arange(N), arg_sn]       # [N]

    agree = (arg_sr == arg_sn)
    disagree = ~agree

    # ---- init outputs ----
    y_pred = np.full((N,), -1, dtype=np.int64)
    method = np.full((N,), "", dtype=object)

    # ---- mode logic ----
    mode = mode.lower()

    if mode == "sr":
        y_pred[:] = arg_sr
        method[:] = "sr"

    elif mode == "sn":
        y_pred[:] = arg_sn
        method[:] = "sn"

    elif mode == "agree":
        y_pred[agree] = arg_sr[agree]
        method[agree] = "agree(sr=sn)"
        # disagreements remain -1 for now; later can fill with original if enabled

    elif mode == "agree_then_sr":
        y_pred[agree] = arg_sr[agree]
        method[agree] = "agree(sr=sn)"
        y_pred[disagree] = arg_sr[disagree]
        method[disagree] = "sr_disagree"

    elif mode == "agree_then_sn":
        y_pred[agree] = arg_sr[agree]
        method[agree] = "agree(sr=sn)"
        y_pred[disagree] = arg_sn[disagree]
        method[disagree] = "sn_disagree"

    elif mode == "agree_then_margin":
        # agree -> accept
        y_pred[agree] = arg_sr[agree]
        method[agree] = "agree(sr=sn)"

        d = np.where(disagree)[0]
        if d.size > 0:
            # per-class or scalar threshold for sr_margin
            if np.isscalar(margin_thr):
                thr = float(margin_thr)
                conf_sr = sr_margin[d] > thr
            else:
                thr = np.asarray(margin_thr, dtype=np.float32)
                assert thr.shape[0] == num_classes
                conf_sr = sr_margin[d] > thr[arg_sr[d]]

            # confident -> sr, else -> sn
            y_pred[d[conf_sr]] = arg_sr[d[conf_sr]]
            method[d[conf_sr]] = "sr_conf"

            y_pred[d[~conf_sr]] = arg_sn[d[~conf_sr]]
            method[d[~conf_sr]] = "sn_fallback"

    else:
        raise ValueError(f"Unknown mode='{mode}'. Use: "
                         f"'sr', 'sn', 'agree', 'agree_then_sr', 'agree_then_sn', 'agree_then_margin'.")

    # ---- fill anything still -1 with original label ----
    if fill_with_original:
        unlabeled = (y_pred < 0)
        y_pred[unlabeled] = y_noisy_all[unlabeled]
        ul = np.where(unlabeled)[0]
        for i in ul:
            method[i] = (method[i] + "|orig") if method[i] else "orig"

    return {
        "idx": idx_all,
        "y_pred": y_pred,
        "method": method,
        "sr_max": sr_max.astype(np.float32),
        "sr_margin": sr_margin.astype(np.float32),
        "sn_min": sn_min.astype(np.float32),
        "y_noisy": y_noisy_all,
        "y_clean": y_clean_all,
        "arg_sr": arg_sr,
        "arg_sn": arg_sn,
        "agree": agree,
    }


def vote_relabel(preds_list, weights=None, vote_thr=0.6, ignore_label=-1):
    P = np.stack(preds_list, axis=0)  # [K, N]
    K, N = P.shape

    y_vote = np.full(N, ignore_label, dtype=np.int64)
    agree_mask = np.zeros(N, dtype=bool)
    agree_frac = np.zeros(N, dtype=np.float32)

    if weights is None:
        for i in range(N):
            col = P[:, i]
            col = col[col != ignore_label]
            if col.size == 0:
                continue
            vals, counts = np.unique(col, return_counts=True)
            j = int(np.argmax(counts))
            y_vote[i] = int(vals[j])

            frac = float(counts[j]) / float(col.size)
            agree_frac[i] = frac
            agree_mask[i] = frac >= vote_thr
        return y_vote, agree_mask, agree_frac

    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    assert weights.shape[0] == K
    for i in range(N):
        cls_scores = {}
        total_w = 0.0
        for k in range(K):
            cls = int(P[k, i])
            if cls == ignore_label:
                continue
            w = float(weights[k])
            total_w += w
            cls_scores[cls] = cls_scores.get(cls, 0.0) + w

        if total_w <= 0 or not cls_scores:
            continue

        best_cls, best_w = max(cls_scores.items(), key=lambda x: x[1])
        y_vote[i] = int(best_cls)

        frac = float(best_w) / float(total_w)
        agree_frac[i] = frac
        agree_mask[i] = frac >= vote_thr

    return y_vote, agree_mask, agree_frac


def relabel_noisy_with_voting(
    *,
    feats_noisy,
    clean_eigs_by_client,
    clean_cids,
    num_classes,
    margin_thr=0.25,
    vote_thr=0.6,
    vote_weights=None,
):
    preds_list = []
    for cid in clean_cids:
        clean_eig = clean_eigs_by_client[cid]
        res = relabel_noisy_samples(
            feats_noisy=feats_noisy,
            clean_eig=clean_eig,
            num_classes=num_classes,
            mode="agree",
            margin_thr=margin_thr,
            fill_with_original=False,
        )
        preds_list.append(np.asarray(res["y_pred"], dtype=np.int64))

    y_vote, agree_mask, agree_frac = vote_relabel(preds_list, weights=vote_weights, vote_thr=vote_thr)

    out = dict(res)
    out["y_pred"] = y_vote
    out["agree_mask"] = agree_mask
    out["agree_frac"] = agree_frac
    return out
