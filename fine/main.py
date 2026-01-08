import torch
import numpy as np
import random
from torch.utils.data import Subset

from dataset import get_dataset, NoisyLabelDataset
from noise import add_noise
from build_model import build_backbone_model
from trainer import TrainConfig, train_model
from fine_detect import (
    extract_backbone_features,
    fine_detect_clean_indices,
    detection_metrics_from_mask,
    per_class_noise_stats, 
    noise_id_metrics,
)

def show_flips(name, idxs, k=10):
    print(f"\n{name} examples (idx | clean -> noisy):")
    for gidx in idxs[:k]:
        print(f"  {gidx:6d} | {int(y_clean[gidx])} -> {int(y_noisy[gidx])}")

if __name__ == "__main__":
    seed = 42
    dataset_name = "cifar10"
    iid = True
    non_iid_prob_class = 0.0
    alpha_dirichlet = 0.0
    num_users = 1

    level_n_system = 1.0
    level_n = 0.5

    model_name = "resnet18"
    pretrained = True
    batch_size = 32

    noise_type="sym"
    save_dir="./figs"

    # -------------------- seeds --------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------- data --------------------
    dataset_train, dataset_test, dict_users, num_classes = get_dataset(
        dataset_name=dataset_name,
        num_users=num_users,
        iid=iid,
        non_iid_prob_class=non_iid_prob_class,
        alpha_dirichlet=alpha_dirichlet,
        seed=seed,
    )

    # keep clean labels BEFORE corruption
    y_clean = np.array(dataset_train.targets, dtype=np.int64)

    # ---- add synthetic client noise (corrupt dataset_train.targets) ----
    y_noisy, gamma_s, real_noise_level, flip_mask, flips_by_client = add_noise(
        seed=seed,
        y_clean=y_clean,
        dict_users=dict_users,
        level_n_system=level_n_system,
        level_n=level_n,
        num_classes=num_classes,
        dataset_name=dataset_name,
        noise_type=noise_type,
        save_dir=save_dir,
    )

    # IMPORTANT: CIFAR expects python list for targets
    dataset_train.targets = y_noisy.tolist()
    # wrap so loader yields (x, noisy_y, idx, clean_y)
    dataset_train_wrap = NoisyLabelDataset(dataset_train, clean_targets=y_clean)

     # ---------- per-client train + FINE ----------
    cfg = TrainConfig(
        epochs=10,          # start small
        lr=1e-3,
        weight_decay=1e-4,
        optimizer="sgd",
    )

    # -------------------- run detection per client --------------------
    for cid in range(num_users):
        sample_idx = np.array(list(dict_users.get(cid, [])), dtype=int)
        if sample_idx.size == 0:
            print(f"[WARN] Client {cid} has 0 samples. Skipping.")
            continue

        # subset the WRAPPED dataset (not the base one)
        dataset_client = Subset(dataset_train_wrap, sample_idx)

        train_loader_client = torch.utils.data.DataLoader(
            dataset=dataset_client,
            batch_size=batch_size,
            shuffle=True,     # training
            num_workers=4,
            persistent_workers=False,
            pin_memory=True,
        )

        eval_loader_client = torch.utils.data.DataLoader(
            dataset=dataset_client,
            batch_size=batch_size,
            shuffle=False,    # feature extraction
            num_workers=4,
            persistent_workers=False,
            pin_memory=True,
        )

        net_local = build_backbone_model(model=model_name, pretrained=pretrained, num_classes=num_classes).to(device)

        net_local, best = train_model(
            model=net_local,
            train_loader=train_loader_client,
            val_loader=None,          # you can pass dataset_test loader if you want
            device=device,
            cfg=cfg,
        )

        feats, y_noisy_c, idx_global, y_clean_c = extract_backbone_features(
            model=net_local,
            loader=eval_loader_client,
            device=device,
        )

        clean_mask, scores = fine_detect_clean_indices(
            feats=feats,
            y_noisy=y_noisy_c,
            num_classes=num_classes,
            zeta=0.5,
        )

        stats = detection_metrics_from_mask(clean_mask, y_noisy_c, y_clean_c)
        overall = noise_id_metrics(clean_mask, y_noisy_c, y_clean_c)
        rows = per_class_noise_stats(clean_mask, y_noisy_c, y_clean_c, num_classes)

        pred_noisy_global_idx = idx_global[~clean_mask]
        true_noisy_global_idx = idx_global[y_noisy_c != y_clean_c]  # from wrapper ground truth

        pred_set = set(pred_noisy_global_idx.tolist())
        true_set = set(true_noisy_global_idx.tolist())
        all_set  = set(idx_global.tolist())

        tp = np.array(sorted(pred_set & true_set), dtype=np.int64)          # predicted noisy AND truly noisy
        fp = np.array(sorted(pred_set - true_set), dtype=np.int64)          # predicted noisy BUT truly clean
        fn = np.array(sorted(true_set - pred_set), dtype=np.int64)          # truly noisy BUT predicted clean
        tn = np.array(sorted((all_set - pred_set) - true_set), dtype=np.int64)  # predicted clean AND truly clean

        # ---------- Overall noise-ID metrics ----------
        print(f"\n[Client {cid}] Overall noise identification (binary)")
        print(f"  Acc={overall['accuracy']:.4f}  Prec={overall['precision']:.4f}  Rec={overall['recall']:.4f}  F1={overall['f1']:.4f}")
        print(f"  TP={overall['tp']}  FP={overall['fp']}  FN={overall['fn']}  TN={overall['tn']}")


        # ---------- Per-class stats ----------
        print("\n  Full per-class table:")
        for r in rows:
            print(
                f"  cls={r['class']:2d} N={r['N']:5d} "
                f"TP={r['TP']:5d} FP={r['FP']:5d} FN={r['FN']:5d} TN={r['TN']:5d} | "
                f"TP%={r['TP_rate']:.3f} FP%={r['FP_rate']:.3f} FN%={r['FN_rate']:.3f} TN%={r['TN_rate']:.3f} | "
                f"Prec={r['precision']:.3f} Rec={r['recall']:.3f} F1={r['f1']:.3f}"
            )