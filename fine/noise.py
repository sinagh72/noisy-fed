# noise.py
import numpy as np
import copy
from plots import (
    plot_transition_heatmap,
    plot_label_distributions,
    label_transition_matrix,
    per_class_flip_summary,
)

CIFAR10_ASYM_MAP = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}  # truck->auto, bird->airplane, cat<->dog, deer->horse


def _sample_sym(label, num_classes, rng):
    candidates = list(range(num_classes))
    candidates.remove(int(label))
    return int(rng.choice(candidates))

def _sample_asym(label, dataset_name, num_classes, rng):
    dataset_name = dataset_name.lower()
    y = int(label)

    if dataset_name == "cifar10":
        return int(CIFAR10_ASYM_MAP.get(y, y))  # if not in map, no flip

    if dataset_name == "cifar100":
        # simple within-group flip (group of 5)
        g = (y // 5) * 5
        return int(g + (y - g + 1) % 5)

    # fallback to symmetric if unknown
    return _sample_sym(y, num_classes, rng)

def add_noise(
    seed,
    y_clean,
    dict_users,
    level_n_system,
    level_n,
    num_classes,
    dataset_name="cifar10",
    noise_type="sym",   # "sym" or "asym"
    save_dir=".",
):
    """
    Adds client-dependent noise.

    Returns:
      y_noisy: np.int64 [N]
      gamma_s: np.int64 [num_users]
      real_noise_level: float [num_users]
      flip_mask: bool [N]  True where label changed
      flips_by_client: dict cid -> np.array of global indices flipped
    """
    rng = np.random.RandomState(seed)
    y_clean = np.asarray(y_clean).astype(int)
    y_noisy = copy.deepcopy(y_clean)

    num_users = len(dict_users)

    gamma_s = rng.binomial(1, level_n_system, num_users)
    gamma_c = np.full(num_users, level_n)

    flip_mask = np.zeros(len(y_clean), dtype=bool)
    real_noise_level = np.zeros(num_users, dtype=float)
    flips_by_client = {}

    for cid in range(num_users):
        sample_idx = np.array(list(dict_users.get(cid, [])), dtype=int)
        if sample_idx.size == 0:
            flips_by_client[cid] = np.array([], dtype=np.int64)
            continue

        if gamma_c[cid] <= 0:
            flips_by_client[cid] = np.array([], dtype=np.int64)
            continue

        prob = rng.rand(len(sample_idx))
        noisy_local = np.where(prob <= gamma_c[cid])[0]

        flipped_global = []
        for j in noisy_local:
            gidx = sample_idx[j]
            orig = int(y_noisy[gidx])

            if noise_type == "sym":
                new = _sample_sym(orig, num_classes, rng)
            elif noise_type == "asym":
                new = _sample_asym(orig, dataset_name, num_classes, rng)
                if new == orig:
                    continue
            else:
                raise ValueError(f"noise_type must be 'sym' or 'asym', got {noise_type}")

            y_noisy[gidx] = new
            if new != orig:
                flip_mask[gidx] = True
                flipped_global.append(gidx)

        noise_ratio = np.mean(y_clean[sample_idx] != y_noisy[sample_idx])
        real_noise_level[cid] = noise_ratio
        flips_by_client[cid] = np.array(flipped_global, dtype=np.int64)

        print(f"Client {cid}: target={gamma_c[cid]:.4f} real={noise_ratio:.4f}  flipped={len(flipped_global)}/{len(sample_idx)}")

    # ----- global plots -----
    # M = label_transition_matrix(y_clean, y_noisy, num_classes)

    # plot_transition_heatmap(
    #     M,
    #     save_path=f"{save_dir}/label_transition_row_norm_{dataset_name}_{noise_type}.png",
    #     normalize="row",
    #     title=f"Label transition ({dataset_name}, {noise_type})",
    # )
    # plot_transition_heatmap(
    #     M,
    #     save_path=f"{save_dir}/label_transition_counts_{dataset_name}_{noise_type}.png",
    #     normalize=None,
    #     title=f"Label transition counts ({dataset_name}, {noise_type})",
    # )
    # plot_label_distributions(
    #     y_before=y_clean,
    #     y_after=y_noisy,
    #     num_classes=num_classes,
    #     save_path=f"{save_dir}/label_distributions_{dataset_name}_{noise_type}.png",
    #     title_prefix=f"Global label distribution ({dataset_name}, {noise_type})",
    # )

    rows = per_class_flip_summary(y_clean, y_noisy, num_classes)

    print("\nPer-class label noise summary:")
    for r in rows:
        print(
            f"  class {r['class']:2d} | "
            f"before={r['total_before']:5d}  "
            f"after={r['total_after_flipped']:5d}  "
            f"flipped_out={r['flipped_out']:5d}  "
            f"flipped_in={r['flipped_in']:5d}  "
            f"delta={r['delta']:+5d}  "
            f"kept={r['kept']:5d}  "
            f"flip_rate={r['flip_rate']:.3f}"
        )

    # np.save(f"{save_dir}/y_clean_{dataset_name}_{noise_type}.npy", y_clean.astype(np.int64))
    # np.save(f"{save_dir}/y_noisy_{dataset_name}_{noise_type}.npy", y_noisy.astype(np.int64))
    # np.save(f"{save_dir}/flipped_indices_{dataset_name}_{noise_type}.npy", np.where(flip_mask)[0])

    return y_noisy, gamma_s, real_noise_level, flip_mask, flips_by_client
