# pipeline.py
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Subset
import copy

from noise import add_noise
from build_model import build_backbone_model
from trainer import TrainConfig, train_model, evaluate, predict_labels_for_indices, extract_backbone_features
from dataset import build_transforms, NoisyLabelDataset
from gmm import fit_mixture
from finesvd_classifier import get_singular_vector, get_score
from plots import print_noise_metrics, correction_metrics
# ----------------------------- seeds -----------------------------
def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    # Ensures each dataloader worker has deterministic numpy/python RNG
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ----------------------------- helpers -----------------------------
def summarize_test(tag: str, stats: dict):
    return (
        f"{tag:10s}: "
        f"loss={stats['loss']:.4f} acc={stats['acc']:.4f} "
        f"prec={stats.get('precision', float('nan')):.4f} "
        f"rec={stats.get('recall', float('nan')):.4f} "
        f"f1={stats.get('f1', float('nan')):.4f} "
        f"auroc={stats.get('auroc', float('nan')):.4f} "
        f"auprc={stats.get('auprc', float('nan')):.4f}"
    )


def make_corrected_labels_subset(y_noisy: np.ndarray, to_change_idx, pred_map):
    y_corr = np.array(y_noisy, copy=True)
    for g in to_change_idx:
        g = int(g)
        if g in pred_map:
            y_corr[g] = int(pred_map[g])
    return y_corr


def _build_loader(ds, batch_size, shuffle, num_workers=4, seed=42, pin_memory=True):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )



# ----------------------------- top-level runner -----------------------------
def run_all_clients(cfg, dataset_train, dataset_test, dict_users, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- clean labels from train ---
    y_clean = np.array(dataset_train.targets, dtype=np.int64)
    # --- inject synthetic noise once globally ---
    y_noisy, *_ = add_noise(
        seed=cfg.seed,
        y_clean=y_clean,
        dict_users=dict_users,
        level_n_system=cfg.level_n_system,
        level_n=cfg.level_n,
        num_classes=num_classes,
        dataset_name=cfg.dataset_name,
        noise_type=cfg.noise_type,
        save_dir=cfg.save_dir,
    )
    y_noisy = np.asarray(y_noisy, dtype=np.int64)

    # --- create noisy dataset ---
    noisy_dataset  = NoisyLabelDataset(dataset_train,  y_noisy=y_noisy, y_clean=y_clean)

    # --- test loader (test dataset already has clean labels only) ---
    test_loader = _build_loader(dataset_test, batch_size=cfg.batch_size, shuffle=False, num_workers=4, seed=cfg.seed)

    # train config
    cfg_train = TrainConfig(
        epochs=getattr(cfg, "epochs", 10),
        lr=getattr(cfg, "lr", 3e-4),
        weight_decay=getattr(cfg, "weight_decay", 1e-2),
        optimizer=getattr(cfg, "optimizer", "AdamW"),
    )

    for cid in range(cfg.num_users):
        run_client_pipeline(
            cfg=cfg,
            device=device,
            noisy_dataset=noisy_dataset,
            test_loader=test_loader,
            dict_users=dict_users,
            num_classes=num_classes,
            cid=cid,
            cfg_train=cfg_train,
            y_clean=y_clean,
            y_noisy=y_noisy,
        )




# ----------------------------- per-client pipeline -----------------------------
def run_client_pipeline(
    cfg,
    device,
    noisy_dataset,
    test_loader,
    dict_users,
    num_classes,
    cid: int,
    cfg_train: TrainConfig,
    y_clean: np.ndarray,
    y_noisy: np.ndarray,
):
    sample_idx = np.array(list(dict_users.get(cid, [])), dtype=int)
    if sample_idx.size == 0:
        print(f"[WARN] Client {cid} has 0 samples. Skipping.")
        return None, None, None

    # Same underlying samples, only label selection changes
    ds_client = Subset(noisy_dataset, sample_idx.tolist())

    plain_base = copy.deepcopy(noisy_dataset.base)
    plain_base.transform = build_transforms(cfg.dataset_name, split="test")
    noisy_dataset_plain = NoisyLabelDataset(plain_base, y_noisy=y_noisy, y_clean=y_clean)
    ds_client_plain = Subset(noisy_dataset_plain, sample_idx.tolist())

    set_all_seeds(cfg.seed)
    base_model = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    del base_model

    train_loader = _build_loader(ds_client, batch_size=cfg.batch_size, shuffle=True, num_workers=4, seed=cfg.seed)
    train_loader_plain = _build_loader(ds_client_plain, batch_size=cfg.batch_size, shuffle=False, num_workers=4, seed=cfg.seed)

    y_noisy_subset = y_noisy[sample_idx]
    y_clean_subset = y_clean[sample_idx]

    print(f"Label identity verified: {np.sum(y_noisy_subset == y_clean_subset)} / {len(y_noisy_subset)}")
    # ================= STAGE 0 =================
    print("\n" + "=" * 90)
    print(f"[STAGE 0] Train on NOISY | client={cid}")
    print("=" * 90)
    net_noisy = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
    net_noisy.load_state_dict(base_state, strict=True)
    net_noisy, _ = train_model(net_noisy, train_loader, None, device, cfg_train, label_mode="noisy")
    test_noisy = evaluate(net_noisy, test_loader, device, num_classes=num_classes, label_mode="clean")
    print("[TEST]", summarize_test("noisy", test_noisy))

    # ================= OPTIONAL STAGE 1/2/3 (kept ready) =================
    # We keep these, but they now use wrap_plain for extraction/inference (deterministic),
    # and we create corrected label arrays, then rebuild wrap_aug with corrected labels.
    # Uncomment when needed.

    # ================= STAGE 1 =================
    print("\n" + "=" * 90)
    print(f"[STAGE 1] Identification | mode={cfg.id_mode} | client={cid}")
    print("=" * 90)
    feats, y_noisy_local, global_idx, y_clean_local = extract_backbone_features(net_noisy, train_loader_plain, device)
    use_prev = False
    prev_features = None
    prev_labels = None
    if use_prev and prev_features is not None and prev_labels is not None:
        vecs = get_singular_vector(prev_features, prev_labels)
    else:
        vecs = get_singular_vector(feats, y_noisy_local)

    scores = get_score(vecs, feats, y_noisy_local, normalization=True)
    y_pred_local = fit_mixture(scores, y_noisy_local, p_threshold=0.5)
    mask_clean = np.zeros((len(y_noisy_local),), dtype=bool)
    mask_clean[y_pred_local] = True
    
    print_noise_metrics("[NOISE IDENTIFICATION]", mask_clean, y_noisy_local, y_clean_local, num_classes)
    pred_noisy_global_idx = global_idx[~mask_clean]
    pred_clean_global_idx = global_idx[mask_clean]
    print(f"[ID] pred_clean={len(pred_clean_global_idx)} pred_noisy={len(pred_noisy_global_idx)}")
    #
    # ================= STAGE 2 =================
    print("\n" + "=" * 90)
    print(f"[STAGE 2] Relabel | strategy={cfg.strategy} | client={cid}")
    print("=" * 90)
    
    teacher = net_noisy
    
    pred_map, conf_map, prob_map = predict_labels_for_indices(
        model=teacher,
        dataloader=train_loader_plain,  
        indices=pred_noisy_global_idx,
        device=device,
        return_probs=True,
    )

    # decide which noisy ones to relabel
    tau = 0.2
    to_change = []
    for g in pred_noisy_global_idx:
        g = int(g)
        if g not in prob_map:
            continue
        conf = float(np.max(prob_map[g]))
        if conf >= tau:
            to_change.append(g)

    to_change = np.asarray(to_change, dtype=np.int64)
    print(f"[RELABEL] flagged_noisy={len(pred_noisy_global_idx)} will_change={len(to_change)} (tau={tau})")

    y_corr = make_corrected_labels_subset(y_noisy, to_change, pred_map)
    cm = correction_metrics(y_clean, y_noisy, y_corr, global_idx=np.array(to_change, dtype=np.int64))
    print("[CORRECTION]", cm)
    #
    # # ================= STAGE 3 =================
    print("\n" + "=" * 90)
    print(f"[STAGE 3] Retrain on corrected | client={cid}")
    print("=" * 90)
    set_all_seeds(cfg.seed)
    corrected_dataset = NoisyLabelDataset(noisy_dataset.base, y_noisy=y_corr, y_clean=y_clean)
    ds_client_corr = Subset(corrected_dataset, sample_idx.tolist())
    train_loader_corr = _build_loader(ds_client_corr, batch_size=cfg.batch_size, shuffle=True, num_workers=4,seed=cfg.seed)
    net_corr = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
    net_corr.load_state_dict(base_state, strict=True)
    net_corr, _ = train_model(net_corr, train_loader_corr, None, device, cfg_train)
    test_corr = evaluate(net_corr, test_loader, device, num_classes=num_classes)
    print("[TEST]", summarize_test("corrected", test_corr))

    # ================= STAGE 4 =================
    print("\n" + "=" * 90)
    print(f"[STAGE 4] ORACLE (clean labels) | client={cid}")
    print("=" * 90)
    set_all_seeds(cfg.seed)
    # Sanity: noise=0 => noisy labels == clean labels for this client
    same_label_rate = float(np.mean(y_noisy[sample_idx] == y_clean[sample_idx]))
    print(f"[SANITY] client noisy==clean rate: {same_label_rate:.3f}")
    train_loader_oracle = _build_loader(ds_client, batch_size=cfg.batch_size, shuffle=True, num_workers=4, seed=cfg.seed)
    net_oracle = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
    net_oracle.load_state_dict(base_state, strict=True)
    net_oracle, _ = train_model(net_oracle, train_loader_oracle, None, device, cfg_train, label_mode="clean")
    test_oracle = evaluate(net_oracle, test_loader, device, num_classes=num_classes, label_mode="clean")
    print("[TEST]", summarize_test("oracle", test_oracle))

    # ================= FINAL =================
    print("\n" + "=" * 90)
    print(f"[FINAL] client={cid}")
    print("=" * 90)
    print(summarize_test("noisy", test_noisy))
    print(summarize_test("corrected", test_corr))
    print(summarize_test("oracle", test_oracle))

  
