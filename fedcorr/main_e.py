import os
import copy
import gc
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from util.options import args_parser
from util.local_training import LocalUpdate, globaltest
from util.fedavg import FedAvg
from util.util import add_full_noise
from util.dataset import get_dataset
from model.build_model import build_backbone_model


# ============================== helpers (Option A: output-space LID) ==============================

def collect_logits(loader, net, device, desc=None, max_points=None):
    """
    Collect logits for samples in loader.
    Returns logits: [N, C] float32 numpy
    """
    net.eval()
    chunks = []
    n_total = 0

    with torch.no_grad():
        it = tqdm(loader, desc=desc, leave=False) if desc else loader
        for images, _ in it:
            images = images.to(device, non_blocking=True)
            logits = net(images)                      # [B, C]
            logits = logits.detach().cpu().float().numpy().astype(np.float32, copy=False)
            chunks.append(logits)
            n_total += logits.shape[0]

            if max_points is not None and n_total >= max_points:
                break

    if len(chunks) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    X = np.concatenate(chunks, axis=0)
    if max_points is not None and X.shape[0] > max_points:
        X = X[:max_points]
    return X


def lid_from_knn_distances(dk, eps=1e-12):
    """
    dk: [N, k] distances to k nearest neighbors (excluding self), sorted ascending.
    returns lids: [N]
    """
    dk = np.asarray(dk, dtype=np.float64)
    v_last = dk[:, -1:] + eps
    lids = -dk.shape[1] / (np.sum(np.log((dk + eps) / v_last), axis=1) + eps)
    return lids.astype(np.float32)


def client_score_lid_mean_from_logits(logits, k=10):
    """
    Option A: LID computed in OUTPUT space, but use logits (not softmax probs).
    - For stability we standardize per-dimension and L2-normalize.
    """
    X = np.asarray(logits, dtype=np.float32)
    n = X.shape[0]
    if n <= k + 1 or X.ndim != 2:
        return 0.0

    # standardize dims (important when logits scale drifts)
    X = X - X.mean(axis=0, keepdims=True)
    X = X / (X.std(axis=0, keepdims=True) + 1e-6)

    # L2 normalize rows (helps for distance geometry)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-12)

    # kNN distances without full NxN matrix
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)

    # drop self-distance at [:,0]
    dk = dists[:, 1:k+1]
    lids = lid_from_knn_distances(dk)
    return float(np.mean(lids))


def pick_clean_cluster_1d(gmm_means):
    """
    For 1D GMM on a 'noisiness' score where higher means more noisy,
    clean cluster is the one with smaller mean.
    """
    return int(np.argmin(gmm_means[:, 0]))


# ============================== main ==============================

if __name__ == "__main__":
    args = args_parser()
    print(args)

    # ---- extra knobs (won't break if not in options.py) ----
    if not hasattr(args, "LID_k"):
        args.LID_k = 10
    if not hasattr(args, "max_points_per_client"):
        args.max_points_per_client = 2000  # cap to avoid runtime blow-up
    if not hasattr(args, "num_workers"):
        args.num_workers = 4

    # seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # (deterministic + benchmark together is contradictory; keep deterministic stable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rootpath = "./record/"
    os.makedirs(rootpath + "txtsave/", exist_ok=True)

    # data
    dataset_train, dataset_test, dict_users = get_dataset(args)

    # ---- add synthetic client noise ----
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = add_full_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    # logging
    txtpath = rootpath + (
        "txtsave/%s_%s_Score_%s_NL_%.2f_LB_%.2f_Iter_%d_ep_%d_AllClients_LR_%.4f_Beta_%.2f_Seed_%d"
        % (args.dataset, args.model, "lid_out_logits",
           args.level_n_system, args.level_n_lowerb,
           args.iteration1, args.local_ep, args.lr, args.beta, args.seed)
    )
    if args.iid:
        txtpath += "_IID"
    else:
        txtpath += "_nonIID_p_%.2f_dirich_%.2f" % (args.non_iid_prob_class, args.alpha_dirichlet)
    if args.correction:
        txtpath += "_CORR"
    if args.mixup:
        txtpath += "_Mix_%.1f" % args.alpha

    f_acc = open(txtpath + "_acc.txt", "a")

    # models (keep on device once)
    netglob = build_backbone_model(args).to(args.device)
    net_local = build_backbone_model(args).to(args.device)

    # sanity: output should be logits [1, C]
    try:
        x0, _ = dataset_train[0]
        x0 = x0.unsqueeze(0).to(args.device)
        with torch.no_grad():
            out0 = netglob(x0)
        print("[Sanity] model output:", tuple(out0.shape))
    except Exception as e:
        print("[WARN] Could not sanity-check model output:", repr(e))

    # score accumulator
    score_accum = np.zeros(args.num_users, dtype=np.float64)

    for iteration in range(args.iteration1):
        print(f"\n========== Preprocess Iteration {iteration+1}/{args.iteration1} ==========")

        # proximal schedule
        if iteration == 0:
            mu_list = np.zeros(args.num_users)
        else:
            mu_list = estimated_noisy_level  # from previous iter

        score_iter = np.zeros(args.num_users, dtype=np.float64)

        w_locals = []
        dict_len = []

        for cid in range(args.num_users):
            sample_idx = np.array(list(dict_users.get(cid, [])), dtype=int)
            if sample_idx.size == 0:
                print(f"[WARN] Client {cid} has 0 samples. Skipping.")
                continue

            dataset_client = Subset(dataset_train, sample_idx)

            # IMPORTANT: persistent_workers=False for short-lived loaders (prevents “gets slower over time”)
            loader_client = torch.utils.data.DataLoader(
                dataset=dataset_client,
                batch_size=args.local_bs,
                shuffle=False,
                num_workers=int(args.num_workers),
                persistent_workers=False,
                pin_memory=True,
            )

            # ---- local train ----
            net_local.load_state_dict(netglob.state_dict())
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx)

            w, loss_epochavg = local.update_weights(
                net=net_local,                 # already on device
                seed=args.seed,
                w_g=netglob,                   # already on device
                epoch=args.local_ep,
                mu=float(mu_list[cid]),
            )

            # store update
            w_locals.append(copy.deepcopy(w))
            dict_len.append(len(sample_idx))

            # eval local (optional)
            net_local.load_state_dict(w)
            acc_t = globaltest(net_local, dataset_test, args)
            f_acc.write(f"iter {iteration}, client {cid}, acc {acc_t:.4f}\n")
            f_acc.flush()

            # ---- Option A scoring: OUTPUT-SPACE LID on logits ----
            logits = collect_logits(
                loader_client,
                net_local,
                args.device,
                desc=f"Client {cid:2d} logits",
                max_points=args.max_points_per_client,
            )

            score_iter[cid] = client_score_lid_mean_from_logits(logits, k=args.LID_k)
            print(f"Client {cid:2d} | score(lid_out_logits) = {score_iter[cid]:.6f}")

            # cleanup to avoid “slows down over time”
            del logits, loader_client, dataset_client
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(w_locals) == 0:
            print("[ERROR] No client updates collected. Exiting.")
            break

        # ---- FedAvg ----
        w_glob = FedAvg(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob))

        score_accum += score_iter

        # ---- GMM ----
        X = score_accum.reshape(-1, 1)
        gmm_client = GaussianMixture(n_components=2, random_state=args.seed).fit(X)
        labels_client = gmm_client.predict(X)

        clean_label = pick_clean_cluster_1d(gmm_client.means_)
        noisy_set = np.where(labels_client != clean_label)[0]
        clean_set = np.where(labels_client == clean_label)[0]

        print("GMM means (lid_out_logits):", gmm_client.means_.flatten())
        print("Detected noisy clients:", noisy_set.tolist())
        print("Detected clean clients:", clean_set.tolist())

        # placeholder for next stages (kept for compatibility)
        estimated_noisy_level = np.zeros(args.num_users, dtype=np.float64)

    print("\nFinal noisy clients:", noisy_set)
    print("Final clean clients:", clean_set)
    f_acc.close()
