# pipeline.py
import numpy as np
import torch
import random
from torch.utils.data import Subset
import copy
import os
from datetime import datetime
from fine.dataset.noise import add_noise
from build_model import build_backbone_model
from trainer import TrainConfig, train_model, predict_all, extract_backbone_features
from dataset.dataset import build_transforms, NoisyLabelDataset
from fine.run import build_loader
from fine.utils.plots_fl import plot_matrix
from relabel import choose_k_by_ratio, relabel_noisy_with_voting
from fine.utils.metrics import per_class_metrics, save_metrics_report_txt, client_identification_metrics
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
def _state_dict_to_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _average_state_dicts(state_dicts):
    """Simple FedAvg (equal weight)."""
    avg = {}
    for k in state_dicts[0].keys():
        vals = [sd[k] for sd in state_dicts]
        if not torch.is_floating_point(vals[0]):
            avg[k] = vals[0]  # keep as-is (or take first)
        else:
            stacked = torch.stack([v.float() for v in vals], dim=0)
            avg[k] = stacked.mean(dim=0).type_as(vals[0])
    return avg



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


def _row_norm(A, eps=1e-12):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / (n + eps)

def compute_eig_from_feats(features):
    eig_dict = {}
    for c, rec in features.items():
        X = rec["feats"].astype(np.float32)  # [Nc, D]
        _, S, v = np.linalg.svd(X, full_matrices=False)
        approx_k = choose_k_by_ratio(S, c=10)
        v_r = v[0].astype(np.float32)
        v_n = v[approx_k:].astype(np.float32)
        v_r = v_r / np.linalg.norm(v_r, axis=0, keepdims=True)
        v_n = v_n / np.linalg.norm(v_n, axis=1, keepdims=True)
        eig_dict[int(c)] = (v_r, v_n)
    return eig_dict



# def compute_eig_from_feats(features):
#     eig_dict = {}
#     for c, rec in features.items():
#         Z = rec["feats"].astype(np.float32)     # [Nc, D]
#         # --- signal direction: mean on sphere ---
#         mu = Z.mean(axis=0)
#         mu_norm = np.linalg.norm(mu) + 1e-12
#         v_s = (mu / mu_norm).astype(np.float32)            # [D]

#         # --- residuals in tangent space (remove mean component) ---
#         proj = (Z @ v_s)[:, None]                           # [Nc,1]
#         R = Z - proj * v_s[None, :]                         # [Nc,D]

#         # R = U S Vt, Vt rows are orthonormal directions in residual space
#         _, S, Vt = np.linalg.svd(R, full_matrices=False)

#         k = choose_k_by_ratio(S, c=10)
#         k = max(1, min(int(k), Vt.shape[0]))

#         # "null" = directions after top-k residual directions, or you can keep top-k residual as "variation"
#         Vn = Vt[k:].astype(np.float32)   # [m,D]

#         eig_dict[int(c)] = (v_s, Vn)
#     return eig_dict


def build_client_loaders(cfg, noisy_dataset, y_noisy, y_clean, dict_users):
    train_loaders_train = {}
    train_loaders_plain = {}

    for cid in range(cfg.num_users):
        sample_idx = np.array(list(dict_users.get(cid, [])), dtype=int).tolist()

        # ---- TRAIN loader: training transforms, shuffle=True
        base_train = copy.deepcopy(noisy_dataset.base)
        base_train.transform = build_transforms(cfg.dataset_name, split="train")
        ds_train = NoisyLabelDataset(base_train, y_noisy=y_noisy, y_clean=y_clean)
        ds_client_train = Subset(ds_train, sample_idx)
        train_loaders_train[cid] = build_loader(
            ds_client_train, batch_size=cfg.batch_size, shuffle=True, num_workers=4, seed=cfg.seed
        )

        # ---- PLAIN loader: "test/plain" transforms, shuffle=False (stable feats)
        base_plain = copy.deepcopy(noisy_dataset.base)
        base_plain.transform = build_transforms(cfg.dataset_name, split="test")
        ds_plain = NoisyLabelDataset(base_plain, y_noisy=y_noisy, y_clean=y_clean)
        ds_client_plain = Subset(ds_plain, sample_idx)
        train_loaders_plain[cid] = build_loader(
            ds_client_plain, batch_size=cfg.batch_size, shuffle=False, num_workers=4, seed=cfg.seed
        )

    return train_loaders_train, train_loaders_plain


def run_n_clients(cfg, dataset_train, dataset_test, dict_users, num_classes, forced_gamma_s=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_dir = f"results/{cfg.noise_type}"

    y_clean = np.array(dataset_train.targets, dtype=np.int64)
    test_loader = build_loader(dataset_test, batch_size=cfg.batch_size, shuffle=False, num_workers=4, seed=cfg.seed)
    # ---- noise injection
    y_noisy, gamma_s, real_noise_level, flip_mask, flips_by_client = add_noise(
        seed=cfg.seed,
        y_clean=y_clean,
        dict_users=dict_users,
        level_n_system=cfg.level_n_system,
        noise_p=cfg.noise_p,
        num_classes=num_classes,
        dataset_name=cfg.dataset_name,
        noise_type=cfg.noise_type,
        forced_gamma_s=forced_gamma_s,   # pass None if you want it sampled
    )
    y_noisy = np.asarray(y_noisy, dtype=np.int64)

    print("\n=== Client noise summary ===")
    for cid in range(cfg.num_users):
        print(f"Client {cid}: gamma_s={int(gamma_s[cid])} real_noise={real_noise_level[cid]:.4f} n={len(dict_users.get(cid, []))}")

    noisy_dataset = NoisyLabelDataset(dataset_train, y_noisy=y_noisy, y_clean=y_clean)

    # ---- per-client loaders
    train_loaders_train, train_loaders_plain = build_client_loaders(cfg, noisy_dataset, y_noisy, y_clean, dict_users)

    # ---- base init
    base_model = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
    base_state = _state_dict_to_cpu(base_model.state_dict())

    # ---- train config (use your TrainConfig if needed)
    cfg_train = TrainConfig(
        epochs=getattr(cfg, "epochs", 10),
        lr=getattr(cfg, "lr", 3e-4),
        weight_decay=getattr(cfg, "weight_decay", 1e-3),
        optimizer=getattr(cfg, "optimizer", "AdamW"),
    )

    eigs_by_client = {}
    similarity_intra_matrix = {}
    null_intra_matrix = {}
    client_stats = {}
    for cid in range(cfg.num_users):
        net = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
        net.load_state_dict(base_state, strict=True)  # same init for fairness
        print(f"\n=== Client {cid} Training ===")
        # ckpt_path = os.path.join(ckpt_path, f"client_{cid}")
        # if os.path.exists(ckpt_path):
        #     net, ckpt = load_model(ckpt_path, net, device)
        # else:
        net, _ = train_model(net, train_loaders_plain[cid], None, device, cfg_train, label_mode="noisy", num_classes=num_classes)
            # save_model(ckpt_path, net, cfg_train=cfg_train,
                # extra={"dataset_name": cfg.dataset_name, "model_name": cfg.model_name, "seed": cfg.seed, "num_classes": num_classes})
        client_stats[cid] = extract_backbone_features(net, train_loaders_plain[cid], device)
        eigs_by_client[cid] = compute_eig_from_feats(client_stats[cid])
        similarity_intra_matrix[cid] = np.ones([num_classes, num_classes], np.float32)
        null_intra_matrix[cid] = np.zeros([num_classes, num_classes], np.float32)
        for i in sorted(eigs_by_client[cid].keys()):
            v_s_i, v_n_i = eigs_by_client[cid][i]
            for j in sorted(eigs_by_client[cid].keys()):
                if i == j:
                    continue
                v_s_j, v_n_j = eigs_by_client[cid][j]
                similarity_intra_matrix[cid][i, j] = np.abs(v_s_i @ v_s_j)
                # A = v_n_i @ v_n_j.T
                # null_intra_matrix[cid][i, j] = (np.linalg.norm(A, ord="fro")**2) / (A.shape[0])

            plot_matrix(input_matrix=similarity_intra_matrix[cid], num_classes=10, save_dir=save_dir, plot_name=f"sim_matrix_{cid}")
            # plot_matrix(input_matrix=null_intra_matrix[cid], num_classes=10, save_dir=save_dir, plot_name=f"null_matrix_{cid}_{r+1}", vmin=np.min(null_intra_matrix[cid]), vmax=np.max(null_intra_matrix[cid]))
        
    X = np.stack([simmat_to_features(similarity_intra_matrix[cid], topk=5) for cid in range(cfg.num_users)], axis=0)
    res = multivariate_gmm_clean_noisy(X, seed=0)

    metrics = client_identification_metrics(gamma_s, res, num_users=cfg.num_users)

    print(f"Identification accuracy: {metrics['acc']*100:.2f}%")
    print(f"TP={metrics['TP']} TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']}")
    print(f"Clean acc (specificity): {metrics['clean_acc']*100:.2f}%")
    print(f"Noisy acc (recall):      {metrics['noisy_acc']*100:.2f}%")
    print("Wrong client IDs:", metrics["wrong_clients"])

    print("Component noisy-scores (higher=noisier):", res["comp_noisy_score"])
    print("Clean clients:", res["clean_idx"].tolist())
    print("Noisy clients:", res["noisy_idx"].tolist())

    acc = metrics["acc"] * 100
    if acc < 100.0:
        log_path = os.path.join(cfg.save_dir, "client_identification_errors.txt")
        os.makedirs(cfg.save_dir, exist_ok=True)

        with open(log_path, "a") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Accuracy: {acc:.2f}%\n")

            f.write(f"TP={metrics['TP']} TN={metrics['TN']} "
                    f"FP={metrics['FP']} FN={metrics['FN']}\n")
            f.write(f"Clean acc: {metrics['clean_acc']*100:.2f}%\n")
            f.write(f"Noisy acc: {metrics['noisy_acc']*100:.2f}%\n")

            f.write(f"Wrong client IDs: {metrics['wrong_clients']}\n")
            f.write(f"Gamma_s: {gamma_s}\n")
            f.write(f"Component noisy scores: {res['comp_noisy_score']}\n")
            f.write(f"Clean clients: {res['clean_idx'].tolist()}\n")
            f.write(f"Noisy clients: {res['noisy_idx'].tolist()}\n")

            f.write(f"Noise type: {cfg.noise_type}\n")
            f.write(f"Noise level: {cfg.level_n}\n")

    
    # optional: rank by suspicion
    rank_noisy = np.argsort(res["p_clean"])   # smallest p_clean = most noisy
    print("Most suspicious (top 10):", rank_noisy.tolist())

    return
    # ---- FedAvg ONLY on clean clients
    clean_cids = [cid for cid in range(cfg.num_users) if int(gamma_s[cid]) == 0]
    noisy_cids = [cid for cid in range(cfg.num_users) if int(gamma_s[cid]) == 1]

    res_dir = f"{cfg.noise_type}/results_{len(noisy_cids)}_noisy.txt"

    if len(clean_cids) == 0:
        print("\n[WARN] No clean clients found. Training ALL clients as noisy (no relabeling).")

        local_states = []
        for cid in noisy_cids:
            net = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
            net.load_state_dict(base_state, strict=True)  # same init for fairness

            print(f"\n=== Noisy Client {cid} Training (no relabel) ===")
            net, _ = train_model(net, train_loaders_train[cid], None, device, cfg_train, label_mode="noisy", num_classes=num_classes)
            local_states.append(_state_dict_to_cpu(net.state_dict()))

        final_state = _average_state_dicts(local_states)

        net_final = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
        net_final.load_state_dict(final_state, strict=True)
        net_final.eval()

        yt, yp = predict_all(net_final, test_loader, device)
        m = per_class_metrics(yt, yp, num_classes=num_classes)
        save_metrics_report_txt(
            os.path.join(cfg.save_dir, res_dir),
            "TEST performance: net_final (FedAvg over all-noisy clients; no clean clients available)",
            m, num_classes,
            extra_lines=[f"num_noisy_clients={len(noisy_cids)}", "relabeling=disabled"],
        )
        return

    print("\n=== FedAvg on clean clients only ===")
    print("Clean clients:", clean_cids)

    local_states = []
    for cid in clean_cids:
        net = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
        net.load_state_dict(base_state, strict=True)
        print(f"\n=== Clean Client {cid} Training ===")
        net, _hist = train_model(net, train_loaders_train[cid], None, device, cfg_train, label_mode="clean", num_classes=num_classes)
        local_states.append(_state_dict_to_cpu(net.state_dict()))

    agg_state = _average_state_dicts(local_states)

    global_net_clean = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
    global_net_clean.load_state_dict(agg_state, strict=True)
    global_net_clean.eval()

    yt, yp = predict_all(global_net_clean, test_loader , device)
    m_clean = per_class_metrics(yt, yp, num_classes=num_classes)
    save_metrics_report_txt(
        os.path.join(cfg.save_dir, res_dir),
        "TEST performance: net_clean (FedAvg over clean clients)",
        m_clean, num_classes
    )


    # ---- compute clean eig per clean client using PLAIN loader (stable features)
    clean_eigs_by_client = {}
    for cid in clean_cids:
        feats_clean_cid = extract_backbone_features(global_net_clean, train_loaders_plain[cid], device)
        clean_eigs_by_client[cid] = compute_eig_from_feats(feats_clean_cid)

    # (optional) weights for voting: e.g. each clean client's acc on its own plain loader
    vote_weights = None  # or list aligned with clean_cids

    noisy_states = []
    sum_kept = 0
    # ---- relabel each noisy client by voting over clean clients' eigens
    for cid in noisy_cids:
        feats_noisy = extract_backbone_features(global_net_clean, train_loaders_plain[cid], device)

        res_vote = relabel_noisy_with_voting(
            feats_noisy=feats_noisy,
            clean_eigs_by_client=clean_eigs_by_client,
            clean_cids=clean_cids,
            num_classes=num_classes,
            margin_thr=0.25,
            vote_thr=0.6,
            vote_weights=vote_weights,
        )

        # report
        keep = res_vote["agree_mask"] & (res_vote["y_pred"] >= 0)
        idx_keep = res_vote["idx"][keep].astype(int)
        y_keep   = res_vote["y_pred"][keep].astype(int)

        allowed_idx_set = set(map(int, idx_keep.tolist()))
        y_override_map  = {int(i): int(y) for i, y in zip(idx_keep, y_keep)}
        print(f"\n=== Noisy client {cid} (VOTING) ===")
        print(f"Kept {keep.sum()} / {keep.size} samples after agreement filter")

        m = per_class_metrics(res_vote["y_clean"][keep], res_vote["y_pred"][keep], num_classes=num_classes)
        print("Overall acc:", m["acc"])
        print("Macro F1:", m["macro"]["f1"])
        print("Micro F1:", m["micro"]["f1"])

        acc_vs_clean = (res_vote["y_pred"][keep] == res_vote["y_clean"][keep]).mean() if keep.any() else float("nan")
        print("Relabel accuracy vs clean (kept only):", acc_vs_clean)

        save_metrics_report_txt(
            os.path.join(cfg.save_dir, res_dir),
            "Relabeling Performance:",
            m, num_classes,
            extra_lines=[f"=== Noisy client {cid} (VOTING) ===", f"Relabel accuracy vs clean (kept only): {acc_vs_clean}"]
        )

        net_noisy = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
        net_noisy.load_state_dict(global_net_clean.state_dict(), strict=True)
        print(f"\n=== Noisy Client {cid} Training ===")
        net_noisy, _ = train_model(
            net_noisy,
            train_loaders_train[cid],   # same loader as before
            None,
            device,
            cfg_train,
            label_mode="noisy",         # base label_mode is irrelevant because we override y
            num_classes=num_classes,
            allowed_idx_set=allowed_idx_set,
            y_override_map=y_override_map,
        )
        noisy_states.append(_state_dict_to_cpu(net_noisy.state_dict()))
        sum_kept += len(allowed_idx_set)


    all_states = local_states + noisy_states
    final_state = _average_state_dicts(all_states)

    net_final = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
    net_final.load_state_dict(final_state, strict=True)
    net_final.eval()

    yt2, yp2 = predict_all(net_final, test_loader, device)
    m_final = per_class_metrics(yt2, yp2, num_classes=num_classes)
    save_metrics_report_txt(
        os.path.join(cfg.save_dir, res_dir),
        "TEST performance: net_final (FedAvg clean + relabeled-noisy models)",
        m_final, num_classes,
        extra_lines=[f"vote_thr=0.6, kept_noisy_total={sum_kept}"]
    )
        

# ----------------------------- top-level runner -----------------------------
# def run_two_clients(cfg, dataset_train, dataset_test, dict_users, num_classes):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     # --- clean labels from train ---
#     y_clean = np.array(dataset_train.targets, dtype=np.int64)
#     ckpt_path = f"./checkpoints/clean_{cfg.dataset_name}_{cfg.model_name}_seed{cfg.seed}.pt"

#     # --- inject synthetic noise once globally ---
#     y_noisy, *_ = add_noise(
#         seed=cfg.seed,
#         y_clean=y_clean,
#         dict_users=dict_users,
#         level_n_system=cfg.level_n_system,
#         level_n=cfg.level_n,
#         num_classes=num_classes,
#         dataset_name=cfg.dataset_name,
#         noise_type=cfg.noise_type,
#     )
#     y_noisy = np.asarray(y_noisy, dtype=np.int64)

#     # --- create noisy dataset ---
#     noisy_dataset  = NoisyLabelDataset(dataset_train,  y_noisy=y_noisy, y_clean=y_clean)

#     # --- test loader (test dataset already has clean labels only) ---
#     test_loader = build_loader(dataset_test, batch_size=cfg.batch_size, shuffle=False, num_workers=4, seed=cfg.seed)

#     # train config
#     cfg_train = TrainConfig(
#         epochs=getattr(cfg, "epochs", 10),
#         lr=getattr(cfg, "lr", 3e-4),
#         weight_decay=getattr(cfg, "weight_decay", 1e-2),
#         optimizer=getattr(cfg, "optimizer", "AdamW"),
#     )


#     base_model = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
#     base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}

#     sample_idx_noisy = np.array(list(dict_users.get(1, [])), dtype=int)
#     ds_client_noisy = Subset(noisy_dataset, sample_idx_noisy.tolist())
#     plain_base_noisy = copy.deepcopy(noisy_dataset.base)
#     plain_base_noisy.transform = build_transforms(cfg.dataset_name, split="test")
#     noisy_dataset_plain = NoisyLabelDataset(plain_base_noisy, y_noisy=y_noisy, y_clean=y_clean)
#     ds_client_plain_noisy = Subset(noisy_dataset_plain, sample_idx_noisy.tolist())
#     train_loader_plain_noisy = build_loader(ds_client_plain_noisy, batch_size=cfg.batch_size, shuffle=False, num_workers=4, seed=cfg.seed)
#     y_noisy_subset_noisy = y_noisy[sample_idx_noisy]
#     y_clean_subset_noisy = y_clean[sample_idx_noisy]

#     print(f"Noisy client Label identity verified: {np.sum(y_noisy_subset_noisy == y_clean_subset_noisy)} / {len(y_noisy_subset_noisy)}")
#     net_noisy = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
#     net_noisy.load_state_dict(base_state, strict=True)
        
#     sample_idx_clean = np.array(list(dict_users.get(0, [])), dtype=int)
#     ds_client_clean = Subset(noisy_dataset, sample_idx_clean.tolist())
#     plain_base_clean = copy.deepcopy(noisy_dataset.base)
#     plain_base_clean.transform = build_transforms(cfg.dataset_name, split="test")
#     noisy_dataset_plain = NoisyLabelDataset(plain_base_clean, y_noisy=y_noisy, y_clean=y_clean)
#     ds_client_plain_clean = Subset(noisy_dataset_plain, sample_idx_clean.tolist())
#     train_loader_plain_clean = build_loader(ds_client_plain_clean, batch_size=cfg.batch_size, shuffle=False, num_workers=4, seed=cfg.seed)
#     y_noisy_subset_clean = y_noisy[sample_idx_clean]
#     y_clean_subset_clean = y_clean[sample_idx_clean]

#     print(f"Clean Client Label identity verified: {np.sum(y_noisy_subset_clean == y_clean_subset_clean)} / {len(y_noisy_subset_clean)}")
#     net_clean = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
#     net_clean.load_state_dict(base_state, strict=True)
#     if os.path.exists(ckpt_path):
#         net_clean, ckpt = load_model(ckpt_path, net_clean, device)
#     else:
#         net_clean, train_hist = train_model(
#             net_clean, train_loader_plain_clean, None, device, cfg_train,
#             label_mode="clean", num_classes=num_classes
#         )
#         save_model(
#             ckpt_path,
#             net_clean,
#             cfg_train=cfg_train,
#             extra={
#                 "dataset_name": cfg.dataset_name,
#                 "model_name": cfg.model_name,
#                 "seed": cfg.seed,
#                 "num_classes": num_classes,
#             },
#         )

#     feats_clean = extract_backbone_features(net_clean, train_loader_plain_clean, device)
#     feats_noisy = extract_backbone_features(net_clean, train_loader_plain_noisy, device)

#     eps = 1e-12
#     quantile = 0.10   # keep ~90% of clean samples as confident

#     clean_eig = {}

#     for c, rec in feats_clean.items():
#         # -----------------------------
#         # 1) get clean features of class c
#         # -----------------------------
#         Xc = rec["feats"].astype(np.float32)      # [Nc, D]
#         # -----------------------------
#         # 2) SVD on clean features
#         # -----------------------------
#         _, S, v = np.linalg.svd(Xc, full_matrices=False)
#         print(f"class {c}: max(S)={S.max():.4f}, min(S)={S.min():.4f}")

#         approx_k = choose_k_by_ratio(S, c=10)
#         idx = np.where(S < 1)[0]
#         v_r = v[0].astype(np.float32)              # principal direction [D]
#         v_n = v[approx_k:].astype(np.float32)      # noise subspace [M_c, D]
#         clean_eig[c] = (v_r, v_n)


#     res = relabel_noisy_samples(
#         feats_noisy=feats_noisy,
#         clean_eig=clean_eig,
#         num_classes=num_classes,
#         mode="agree",
#         margin_thr=0.25,
#     )

#     m = per_class_metrics(res["y_clean"], res["y_pred"], num_classes=num_classes)

#     print("Overall acc:", m["acc"])
#     print("Macro F1:", m["macro"]["f1"])
#     print("Micro F1:", m["micro"]["f1"])  # equals acc in multiclass single-label

#     for c in range(num_classes):
#         print(
#             f"Class {c}: "
#             f"P={m['precision'][c]:.3f}, R={m['recall'][c]:.3f}, F1={m['f1'][c]:.3f}, "
#             f"TP={m['TP'][c]}, FP={m['FP'][c]}, FN={m['FN'][c]}, TN={m['TN'][c]}, "
#             f"support={m['support'][c]}"
#         )


#     # quick sanity stats if you have y_clean 
#     acc_vs_clean = (res["y_pred"] == res["y_clean"]).mean()
#     print("Relabel accuracy vs clean:", acc_vs_clean)

    # clean_eig = {}
    # tol = 1e-6  # or relative tolerance: tol = 1e-6 * S[0]

    # for c, rec in feats_clean.items():
    #     Xc = rec["feats"].astype(np.float32)
    #     # Xc_mean = np.mean(Xc, axis=0)
    #     _, S, v = np.linalg.svd(Xc)
    #     # clean_eig[c] =  v[0].astype(np.float32)
    #     print(max(S), min(S))
    #     idxs = np.where(S < 1)[0]
    #     print(idxs.shape)
    #     approx_k = choose_k_by_ratio(S, c=10)
    #     # clean_eig[c] = (v[0].astype(np.float32), v[zero_idx].astype(np.float32))
    #     print(approx_k)
    #     clean_eig[c] = v[approx_k:].astype(np.float32)
    #     # clean_mean[c] =  Xc_mean / (np.linalg.norm(Xc_mean) + 1e-12)

    # separation_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)

    # for c, rec in feats_clean.items():
    #     Xc = rec["feats"].astype(np.float32)
    #     for j in range(num_classes):
    #         vj = clean_eig[j]
    #         print(Xc.shape)
    #         print(vj.shape)
    #         separation_matrix[c,j] = np.linalg.norm(Xc @ vj.T, "fro") / np.sqrt(Xc.shape[0])

    # separation_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    # for i in range(num_classes):
    #     for j in range(num_classes):
    #         vi = clean_eig[i]
    #         vj = clean_eig[j]
    #         rel = float(np.abs(np.dot(vi, vj)))
    #         # rel = float(np.linalg.norm(np.dot(vi, vj.T)))
    #         separation_matrix[i, j] = rel
    # # --- plot heatmap with values ---
    # plt.figure(figsize=(7, 6))
    # im = plt.imshow(separation_matrix)
    # plt.colorbar(im)

    # plt.title("Inter-class Separation Matrix (Principal Directions)")
    # plt.xlabel("Class j")
    # plt.ylabel("Class i")

    # plt.xticks(range(num_classes))
    # plt.yticks(range(num_classes))

    # # # annotate each cell
    # for i in range(num_classes):
    #     for j in range(num_classes):
    #         value = separation_matrix[i, j]
    #         plt.text(
    #             j, i,
    #             f"{value:.2f}",
    #             ha="center",
    #             va="center",
    #             color="white" if value > separation_matrix.max() * 0.5 else "black",
    #             fontsize=9
    #         )

    # plt.tight_layout()
    # plt.savefig("separation_matrix.png")
    # plt.close()

    

    # for c in range(10):
    #     separation_matrix_noisy = np.zeros((feats_noisy[c]["feats"].shape[0], 10), dtype=np.float32)
    #     print(feats_noisy[c]["feats"].shape)
    #     for z, j in enumerate(range(10)):
    #         for i, feat in enumerate(feats_noisy[c]["feats"]):
    #             vj = clean_eig[j]
    #             # rel = float(np.abs(np.linalg.norm(np.dot(vj, feat))))
    #             rel = float(np.abs(np.dot(vj, feat)) / (np.linalg.norm(feat) + 1e-12))
    #             separation_matrix_noisy[i, z] = rel
    #     scores_j0 = separation_matrix_noisy[:, 0]
    #     scores_j1 = separation_matrix_noisy[:, 1]
    #     scores_j2 = separation_matrix_noisy[:, 2]
    #     scores_j3 = separation_matrix_noisy[:, 3]
    #     scores_j4 = separation_matrix_noisy[:, 4]
    #     scores_j5 = separation_matrix_noisy[:, 5]
    #     scores_j6 = separation_matrix_noisy[:, 6]
    #     scores_j7 = separation_matrix_noisy[:, 7]
    #     scores_j8 = separation_matrix_noisy[:, 8]
    #     scores_j9 = separation_matrix_noisy[:, 9]


    #     plt.figure(figsize=(7, 5))

    #     plt.hist(scores_j0, bins=50, alpha=0.6, label="Class 0 eig direction", density=True)
    #     plt.hist(scores_j1, bins=50, alpha=0.6, label="Class 1 eig direction", density=True)
    #     plt.hist(scores_j2, bins=50, alpha=0.6, label="Class 2 eig direction", density=True)
    #     plt.hist(scores_j3, bins=50, alpha=0.6, label="Class 3 eig direction", density=True)
    #     plt.hist(scores_j4, bins=50, alpha=0.6, label="Class 4 eig direction", density=True)
    #     plt.hist(scores_j5, bins=50, alpha=0.6, label="Class 5 eig direction", density=True)
    #     plt.hist(scores_j6, bins=50, alpha=0.6, label="Class 6 eig direction", density=True)
    #     plt.hist(scores_j7, bins=50, alpha=0.6, label="Class 7 eig direction", density=True)
    #     plt.hist(scores_j8, bins=50, alpha=0.6, label="Class 8 eig direction", density=True)
    #     plt.hist(scores_j9, bins=50, alpha=0.6, label="Class 9 eig direction", density=True)

    #     plt.xlabel("Cosine similarity |⟨v_j, noisy feat⟩|")
    #     plt.ylabel("Density")
    #     plt.title("Noisy Feature Projection Distributions")
    #     plt.legend()

    #     plt.tight_layout()
    #     plt.savefig(f"noisy_feature_projections_{c}.png")
    #     plt.close()

    #     mean_0 = separation_matrix_noisy[:, c].mean()

    #     print(f"Mean projection onto class {c} eig direction: {mean_0:.4f}")

    #     total_diff = 0.0
    #     # means for other classes w.r.t. class 0
    #     for j in range(1, separation_matrix_noisy.shape[1]):
    #         mean_j = separation_matrix_noisy[:, j].mean()
    #         diff = mean_0 - mean_j
    #         total_diff += diff
    #         print(
    #             f"Class {c} vs Class {j}: "
    #             f"mean_j={mean_j:.4f}, "
    #             f"Δ({c}−{j})={diff:.4f}"
    #         )
    #     print(f"\nTOTAL separation (Class {c} vs all others): {total_diff:.4f}")
    # K = num_classes
    # means_mat = np.stack([clean_mean[c] for c in range(K)], axis=0).astype(np.float32)
    
    # N_noisy = len(sample_idx_noisy)
    # matrix_res = np.zeros((N_noisy, K), dtype=np.float32)
    # idx_to_row = {int(gidx): r for r, gidx in enumerate(sample_idx_noisy)}

    # for _, rec in feats_noisy.items():
    #     Xn = rec["feats"].astype(np.float32)  # [Nn,D]
    #     idx = rec["idx"]

    #     Xn_hat = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12)  # [Nn,D]
    #     scores_block = np.abs(Xn_hat @ means_mat.T)  # [Nn,K]

    #     for j in range(len(idx)):
    #         r = idx_to_row[int(idx[j])]
    #         matrix_res[r, :] = scores_block[j, :]

    # y_pred = np.argmax(matrix_res, axis=1).astype(np.int64)

    # y_true = y_clean_subset_noisy.astype(np.int64)
    
    # acc = (y_pred == y_true).mean()
    # print(f"Argmax classifier accuracy vs y_true: {acc:.4f} ({(y_pred==y_true).sum()}/{len(y_true)})")

    # K = matrix_res.shape[1]
    # cm = np.zeros((K, K), dtype=np.int64)
    # for t, p in zip(y_true, y_pred):
    #     cm[t, p] += 1
    # print("Confusion matrix (rows=true, cols=pred):\n", cm)


    #     eig_res_clean[c] = {
    #     "vectors": vectors.astype(np.float32),   # full spectrum (optional)
    #     "v_top": v_top,                         # [D] principal direction
    # }

    # class_ids = list(range(10))
    # plot_pca_mean_vs_vtop(feats_clean, eig_res_clean, class_ids=class_ids)
