# fit_gmm.py
import numpy as np
from sklearn.mixture import GaussianMixture

def simmat_to_features(M: np.ndarray, topk=10):
    """
    Unified features for SYM + ASYM.
    Returns features where larger values typically indicate 'noisier'.
    """
    M = np.asarray(M, dtype=np.float64)
    C = M.shape[0]
    M = np.clip(M, 0.0, 1.0)

    off = M[~np.eye(C, dtype=bool)]  # [C*(C-1)]
    if off.size == 0:
        return np.zeros(8, dtype=np.float64)

    off_sorted = np.sort(off)
    k = min(topk, off_sorted.size)

    # level + dispersion
    mean = off.mean()
    std  = off.std()
    cv   = std / (mean + 1e-12)

    # tail / peak stats
    topk_mean = off_sorted[-k:].mean()
    p90 = np.quantile(off, 0.90)
    p95 = np.quantile(off, 0.95)
    p99 = np.quantile(off, 0.99)

    # per-row max: "one strong confusion per class" (asym signature)
    row_max = []
    for i in range(C):
        row = np.delete(M[i], i)
        row_max.append(row.max() if row.size else 0.0)
    row_max = np.asarray(row_max, dtype=np.float64)
    row_max_mean = row_max.mean()

    return np.array([mean, std, cv, topk_mean, p90, p95, p99, row_max_mean], dtype=np.float64)


def multivariate_gmm_clean_noisy(X: np.ndarray, seed: int = 0):
    """
    X: [N_clients, D] unified features (higher => noisier).
    Returns clean/noisy split via 2-GMM.
    """
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    if N < 2:
        raise ValueError("Need >=2 clients.")
    if np.allclose(X, X[0]):
        clean_mask = np.ones(N, dtype=bool)
        return {"p_clean": np.ones(N), "clean_idx": np.arange(N), "noisy_idx": np.array([], dtype=int)}

    # standardize
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True) + 1e-12
    Z = (X - mu) / sig

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="diag",
        reg_covar=1e-6,
        n_init=30,
        max_iter=1000,
        random_state=seed,
    ).fit(Z)

    # Feature order: [mean, std, cv, topk_mean, p90, p95, p99, row_max_mean]
    means2 = gmm.means_  # [2,D] in standardized space

    # Robust "noisiness" score for each component:
    # - mean/p95 capture symmetric lift
    # - p99/topk/cv/row_max capture asymmetric spikes
    comp_noisy_score = (
        0.6 * means2[:, 0] +   # mean
        0.4 * means2[:, 5] +   # p95
        0.7 * means2[:, 6] +   # p99
        0.5 * means2[:, 3] +   # topk_mean
        0.5 * means2[:, 2] +   # cv
        0.5 * means2[:, 7]     # row_max_mean
    )
    # mean:
    # High → many classes are somewhat confused
    # Low → classes are mostly well separated

    # std:
    # Low → similarities are uniform
    # High → some pairs much worse than others
    
    # cv
    # Low → confusion evenly spread 
    # High → a few class pairs dominate the confusion


    #topk_mean = average of the largest K similarities
    #Focuses only on worst confusions

    noisy_comp = int(np.argmax(comp_noisy_score))
    clean_comp = 1 - noisy_comp

    p_clean = gmm.predict_proba(Z)[:, clean_comp]
    clean_mask = p_clean >= 0.5

    return {
        "p_clean": p_clean,
        "clean_idx": np.where(clean_mask)[0],
        "noisy_idx": np.where(~clean_mask)[0],
        "comp_noisy_score": comp_noisy_score,
        "clean_comp": clean_comp,
        "noisy_comp": noisy_comp,
        "gmm": gmm,
    }
