# run.py
import copy, os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from fine.dataset.noise import add_noise, add_noise_2
from build_model import build_backbone_model
from dataset.dataset import build_transforms, NoisyLabelDataset
from fine.strategy.jal_strategy import run_jal
from fine.strategy.eig_jal_strategy import run_eig_jal_strategy
from trainer import TrainConfig
from fine.strategy.baseline_strategy import run_fedavg, state_dict_to_cpu
from fine.strategy.eig_strategy import run_eig_strategy
from fine.strategy.pruning_strategy import run_pruning_strategy
from utils.set_seed import seed_worker, make_generator
from utils.plots_fl import plot_client_distributions, save_aug_sanity_grid
from dataset.sampling import build_balanced_aug_loader

def build_loader(ds, batch_size, shuffle, num_workers=4, seed=42, pin_memory=True, drop_last=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )


def build_client_loaders(cfg, noisy_dataset, y_noisy, y_clean, dict_users, num_classes, extra_aug=None):
    train_loaders_train = {}
    balanced_train_loaders = {}
    train_loaders_plain = {}


    for cid in range(cfg.num_users):
        sample_idx = np.array(list(dict_users.get(cid, [])), dtype=int).tolist()

        # ---- TRAIN loader: training transforms, shuffle=True
        base_train = copy.deepcopy(noisy_dataset.base)
        base_train.transform = build_transforms(cfg.dataset_name, split="train")
        ds_train = NoisyLabelDataset(base_train, y_noisy=y_noisy, y_clean=y_clean)
        ds_client_train = Subset(ds_train, sample_idx)
        train_loaders_train[cid] = build_loader(ds_client_train, batch_size=cfg.batch_size, shuffle=True, num_workers=4, seed=cfg.seed, drop_last=False)
        if "fedeig" in cfg.fl_strategy:
            balanced_train_loaders[cid], info = build_balanced_aug_loader(
                ds_client_train,
                num_classes=num_classes,
                batch_size=cfg.batch_size,
                seed=cfg.seed + cid,
                balance_on="noisy",              # or "clean"
                target_per_class="max",
                cap_target=getattr(cfg, "balance_cap_target", None),
                aug_transform=extra_aug,         # make sure this is transform-safe (see warning above)
            )
            print(f"[cid {cid}] present={len(info['present'])} target={info['target']} added={info['added']} final_len={info['final_len']}")
        else:
            balanced_train_loaders[cid] = None
        # ---- PLAIN loader: "test/plain" transforms, shuffle=False (stable feats)
        base_plain = copy.deepcopy(noisy_dataset.base)
        base_plain.transform = build_transforms(cfg.dataset_name, split="test")
        ds_plain = NoisyLabelDataset(base_plain, y_noisy=y_noisy, y_clean=y_clean)
        ds_client_plain = Subset(ds_plain, sample_idx)
        train_loaders_plain[cid] = build_loader(ds_client_plain, batch_size=cfg.batch_size, shuffle=False, num_workers=4, seed=cfg.seed)

    return train_loaders_train, balanced_train_loaders, train_loaders_plain

def run_fl(cfg, dataset_train, dataset_test, dict_users, num_classes, forced_gamma_s=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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
        forced_gamma_s=forced_gamma_s,   
    )
    # y_noisy, gamma_s, real_noise_level, flip_mask, flips_by_client = add_noise_2(
    #     seed=cfg.seed,
    #     y_clean=y_clean,
    #     dict_users=dict_users,
    #     level_n_system=cfg.level_n_system,
    #     eta=cfg.noise_p,
    #     num_classes=num_classes,
    #     dataset_name=cfg.dataset_name,
    #     noise_type=cfg.noise_type,
    #     gamma_s=forced_gamma_s,   
    # )
    y_noisy = np.asarray(y_noisy, dtype=np.int64)
    distribution = "" if cfg.iid else f"_Non-IID_{cfg.alpha_dirichlet}"
    save_dir = f"{cfg.save_dir}/{cfg.noise_type}{distribution}/{gamma_s.tolist().count(0)} clean/{cfg.noise_p}/{cfg.fl_strategy}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("\n=== Client noise summary ===")
    for cid in range(cfg.num_users):
        print(f"Client {cid}: gamma_s={int(gamma_s[cid])} real_noise={real_noise_level[cid]:.4f} n={len(dict_users.get(cid, []))}")

    noisy_dataset = NoisyLabelDataset(dataset_train, y_noisy=y_noisy, y_clean=y_clean)
    # ---- per-client loaders
    import torchvision.transforms as T
    extra_aug = T.Compose([
        T.RandomErasing(p=0.7, scale=(0.02, 0.25), ratio=(0.3, 3.3), value="random"),
        T.Lambda(lambda x: x + 0.03 * torch.randn_like(x)),
    ])
    train_loaders_train, balanced_train_loaders, train_loaders_plain = build_client_loaders(cfg, noisy_dataset, y_noisy, y_clean, dict_users, num_classes, extra_aug=extra_aug)
    if "fedeig" in cfg.fl_strategy:
        plot_dir = os.path.join(save_dir, "class_dists")
        plot_client_distributions(
            dict_users=dict_users,
            num_classes=num_classes,
            train_loaders=train_loaders_train,
            balanced_loaders=balanced_train_loaders,
            label_pos=1,             # your batch is (x, y_noisy, y_clean, idx)
            max_batches=500,
            mode="count",       # or "count"
            save_dir=plot_dir,
            prefix="train",
        )
        # vis_dir = os.path.join(save_dir, "aug_sanity")
        # for cid in range(cfg.num_users):
        #     save_aug_sanity_grid(train_loaders_train[cid], os.path.join(vis_dir, f"cid_{cid}_base.png"))
        #     save_aug_sanity_grid(balanced_train_loaders[cid], os.path.join(vis_dir, f"cid_{cid}_balanced_aug.png"))
        
    # ---- base init
    base_model = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
    base_state = state_dict_to_cpu(base_model.state_dict())

    # ---- train config (use your TrainConfig if needed)
    cfg_train = TrainConfig(epochs=getattr(cfg, "epochs", 10),
                            lr=getattr(cfg, "lr", 3e-4),
                            weight_decay=getattr(cfg, "weight_decay", 1e-2),
                            optimizer=getattr(cfg, "optimizer", "AdamW"))

    cfg_identification = TrainConfig(epochs=getattr(cfg, "identification_epochs", 10),
                            lr=getattr(cfg, "lr", 3e-4),
                            weight_decay=getattr(cfg, "weight_decay", 1e-2),
                            optimizer=getattr(cfg, "optimizer", "AdamW"))
    

    if cfg.fl_strategy == "fedavg":
        run_fedavg(cfg=cfg, gamma_s=gamma_s, num_classes=num_classes, base_state=base_state, train_loaders_train=train_loaders_train, test_loader=test_loader,
                   cfg_train=cfg_train, save_dir=save_dir, device=device)


    elif cfg.fl_strategy == "fedprox": #TODO: implement FedProx
        run_fedavg(cfg=cfg, gamma_s=gamma_s, num_classes=num_classes, base_state=base_state, train_loaders_train=train_loaders_train, test_loader=test_loader,
            cfg_train=cfg_train, save_dir=save_dir, device=device)
    
    elif cfg.fl_strategy == "jal":
        run_jal(cfg=cfg, gamma_s=gamma_s, num_classes=num_classes, base_state=base_state, train_loaders_train=train_loaders_train, test_loader=test_loader,
                   cfg_train=cfg_train, save_dir=save_dir, device=device)
        
    elif cfg.fl_strategy == "fedeig": 
        run_eig_strategy(cfg=cfg, gamma_s=gamma_s, num_classes=num_classes, base_state=base_state, train_loaders_train=train_loaders_train,
                          test_loader=test_loader, train_loaders_plain=train_loaders_plain, cfg_train=cfg_train, balanced_train_loaders=balanced_train_loaders,
                          cfg_identification=cfg_identification, save_dir=save_dir, device=device)
        
    elif cfg.fl_strategy == "fedeigjal": 
        run_eig_jal_strategy(cfg=cfg, gamma_s=gamma_s, num_classes=num_classes, base_state=base_state, train_loaders_train=train_loaders_train,
                          test_loader=test_loader, train_loaders_plain=train_loaders_plain, cfg_train=cfg_train, 
                          cfg_identification=cfg_identification, save_dir=save_dir, device=device)
         
    # elif cfg.fl_strategy == "fedeigkd": 
    #     run_eig_kd_strategy(cfg=cfg, gamma_s=gamma_s, num_classes=num_classes, base_state=base_state, train_loaders_train=train_loaders_train,
    #                         test_loader=test_loader, train_loaders_plain=train_loaders_plain, cfg_train=cfg_train, 
    #                         cfg_identification=cfg_identification, save_dir=save_dir, device=device)    
        
    elif cfg.fl_strategy == "pruning": 
        run_pruning_strategy(cfg=cfg, gamma_s=gamma_s, num_classes=num_classes, base_state=base_state, train_loaders_train=train_loaders_train,
                            test_loader=test_loader, cfg_train=cfg_train, cfg_identification=cfg_identification, save_dir=save_dir, device=device)