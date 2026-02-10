
# eig_strategy.py
import numpy as np
import os
from datetime import datetime
from build_model import build_backbone_model
from trainer import train_model, predict_all, extract_backbone_features
from relabel import choose_k_by_ratio, relabel_noisy_with_voting
from fine.utils.metrics import per_class_metrics, save_metrics_report_txt, client_identification_metrics
from fine.utils.plots_fl import plot_matrix
from strategy.fit_gmm import multivariate_gmm_clean_noisy, simmat_to_features
from strategy.baseline_strategy import average_state_dicts, state_dict_to_cpu


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


def run_eig_strategy(cfg, gamma_s, num_classes, base_state, train_loaders_train, train_loaders_plain, test_loader, cfg_train, save_dir, device):
    eigs_by_client = {}
    similarity_intra_matrix = {}
    null_intra_matrix = {}
    client_stats = {}
    # ===================================== clean client identification =================================================
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

    
    
    rank_noisy = np.argsort(res["p_clean"])   # smallest p_clean = most noisy
    print("Most suspicious (top 10):", rank_noisy.tolist())

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
            local_states.append(state_dict_to_cpu(net.state_dict()))

        final_state = average_state_dicts(local_states)

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
        local_states.append(state_dict_to_cpu(net.state_dict()))

    agg_state = average_state_dicts(local_states)

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
        noisy_states.append(state_dict_to_cpu(net_noisy.state_dict()))
        sum_kept += len(allowed_idx_set)


    all_states = local_states + noisy_states
    final_state = average_state_dicts(all_states)

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
