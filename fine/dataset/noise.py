# noise.py
import numpy as np
import copy
from fine.utils.plots import (
    plot_transition_heatmap,
    plot_label_distributions,
    label_transition_matrix,
)

CIFAR10_ASYM_MAP = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}  # truck->auto, bird->airplane, cat<->dog, deer->horse
# CIFAR10_ASYM_MAP = {3: 5, 4: 7, 9: 1, 8: 9, 0: 8, 2:0, 6:4}  # truck->auto, bird->airplane, cat<->dog, deer->horse


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


def per_class_flip_summary(y_before, y_after, num_classes, sample_idx):
    y_before = np.asarray(y_before, dtype=np.int64)[sample_idx]
    y_after  = np.asarray(y_after,  dtype=np.int64)[sample_idx]

    before_counts = np.bincount(y_before, minlength=num_classes)
    after_counts  = np.bincount(y_after,  minlength=num_classes)

    changed = (y_before != y_after)
    flipped_out = np.bincount(y_before[changed], minlength=num_classes)
    flipped_in  = np.bincount(y_after[changed],  minlength=num_classes)

    kept = before_counts - flipped_out

    rows = []
    for c in range(num_classes):
        total_before = int(before_counts[c])
        total_after  = int(after_counts[c])
        out_c = int(flipped_out[c])
        in_c  = int(flipped_in[c])
        kept_c = int(kept[c])
        delta = total_after - total_before
        flip_rate = (out_c / total_before) if total_before > 0 else 0.0

        rows.append({
            "class": c,
            "total_before": total_before,
            "total_after_flipped": total_after,
            "flipped_out": out_c,
            "flipped_in": in_c,
            "delta": delta,
            "kept": kept_c,
            "flip_rate": float(flip_rate),
        })
    return rows


def add_noise(
    seed,
    y_clean,
    dict_users,
    level_n_system,     # P(client is noisy)
    noise_p,            # target flip rate inside noisy clients
    num_classes,
    dataset_name="cifar10",
    noise_type="sym",   # "sym" or "asym"
    max_asym_rounds=10, # for asym: retries to hit target flips
    forced_gamma_s=None,
):
    """
    Client-dependent label noise with strict per-client targets:
      - If gamma_s[cid]==0: 0 flips guaranteed.
      - If gamma_s[cid]==1: flip exactly k=round(noise_p * n_client) labels
        (or as close as possible for asym if mapping makes target impossible).

    Returns:
      y_noisy, gamma_s, real_noise_level, flip_mask, flips_by_client
    """
    rng = np.random.RandomState(seed)
    y_clean = np.asarray(y_clean, dtype=np.int64)
    y_noisy = y_clean.copy()

    num_users = len(dict_users)
    if forced_gamma_s is None:
        # gamma_s = rng.binomial(1, level_n_system, size=num_users).astype(np.int64)
        k = int(round(level_n_system * num_users))
        k = max(0, min(num_users, k))

        gamma_s = np.zeros(num_users, dtype=np.int64)
        noisy_ids = rng.choice(num_users, size=k, replace=False)
        gamma_s[noisy_ids] = 1

    else:
        gamma_s = np.asarray(forced_gamma_s, dtype=np.int64)
        assert gamma_s.shape[0] == num_users


    flip_mask = np.zeros(len(y_clean), dtype=bool)
    real_noise_level = np.zeros(num_users, dtype=float)
    flips_by_client = {}

    for cid in range(num_users):
        sample_idx = np.asarray(list(dict_users.get(cid, [])), dtype=np.int64)
        n = len(sample_idx)

        if n == 0:
            flips_by_client[cid] = np.array([], dtype=np.int64)
            real_noise_level[cid] = 0.0
            print(f"Client {cid}: EMPTY")
            continue

        # ----------------- CLEAN CLIENT => GUARANTEED 0 NOISE -----------------
        if gamma_s[cid] == 0 or noise_p <= 0:
            flips_by_client[cid] = np.array([], dtype=np.int64)
            real_noise_level[cid] = 0.0
            # sanity check: truly clean
            assert np.all(y_noisy[sample_idx] == y_clean[sample_idx])
            print(f"Client {cid}: CLEAN (gamma_s=0)  real=0.0000  flipped=0/{n}")
            continue

        # ----------------- NOISY CLIENT => TARGET EXACT #FLIPS -----------------
        if noise_type == "asym":
            # flippable = labels that actually change under the asym map
            flippable_mask = np.isin(y_noisy[sample_idx], list(CIFAR10_ASYM_MAP.keys()))
            flippable_idx = sample_idx[flippable_mask]
            n_flippable = int(flippable_idx.size)

            # If nothing flippable, this client cannot receive asym noise
            if n_flippable == 0:
                flips_by_client[cid] = np.array([], dtype=np.int64)
                real_noise_level[cid] = 0.0  # "asym flip rate among flippables" is 0
                print(f"Client {cid}: ASYM but NO FLIPPABLE labels -> flipped=0/{n} (flippable=0)")
                continue
            # target flips defined over flippables, not all samples
            k_target = int(np.round(noise_p * n_flippable))
            k_target = max(0, min(k_target, n_flippable))
        else:
            # symmetric uses all samples
            k_target = int(np.round(noise_p * n))
            k_target = max(0, min(k_target, n))

        flipped_global = []

        if noise_type == "sym":
            # EXACT: choose k_target indices and always flip
            chosen = rng.choice(sample_idx, size=k_target, replace=False)
            for gidx in chosen:
                gidx = int(gidx)
                orig = int(y_noisy[gidx])
                new = _sample_sym(orig, num_classes, rng)  # should guarantee new != orig
                if new == orig:
                    # defensive: if _sample_sym is buggy, resample a few times
                    for _ in range(5):
                        new = _sample_sym(orig, num_classes, rng)
                        if new != orig:
                            break
                if new == orig:
                    continue
                y_noisy[gidx] = int(new)
                flip_mask[gidx] = True
                flipped_global.append(gidx)

            # If _sample_sym is correct, len(flipped_global) == k_target
            # If not, we may be short; fix by topping up:
            if len(flipped_global) < k_target:
                remaining = k_target - len(flipped_global)
                pool = np.setdiff1d(sample_idx, np.asarray(flipped_global, dtype=np.int64), assume_unique=False)
                if remaining > 0 and pool.size > 0:
                    extra = rng.choice(pool, size=min(remaining, pool.size), replace=False)
                    for gidx in extra:
                        gidx = int(gidx)
                        orig = int(y_noisy[gidx])
                        new = _sample_sym(orig, num_classes, rng)
                        if new != orig:
                            y_noisy[gidx] = int(new)
                            flip_mask[gidx] = True
                            flipped_global.append(gidx)

        elif noise_type == "asym":
            # Asym may not be possible for some labels; we retry until we hit k_target or exhaust.
            remaining = k_target
            used = set()

            # We repeatedly sample from remaining pool and keep only real flips.
            for _round in range(max_asym_rounds):
                if remaining <= 0:
                    break

                pool = [int(g) for g in flippable_idx  if int(g) not in used]
                if not pool:
                    break

                # draw a batch of candidates
                draw = min(len(pool), max(remaining * 5, remaining))
                cand = rng.choice(np.asarray(pool, dtype=np.int64), size=draw, replace=False)

                for gidx in cand:
                    if remaining <= 0:
                        break
                    gidx = int(gidx)
                    used.add(gidx)

                    orig = int(y_noisy[gidx])
                    new = _sample_asym(orig, dataset_name, num_classes, rng)
                    if new == orig:
                        continue

                    y_noisy[gidx] = int(new)
                    flip_mask[gidx] = True
                    flipped_global.append(gidx)
                    remaining -= 1

            # If we fail to reach k_target, it's because asym mapping couldn't flip enough samples.

        else:
            raise ValueError(f"noise_type must be 'sym' or 'asym', got {noise_type}")

        flips_by_client[cid] = np.asarray(flipped_global, dtype=np.int64)

        noise_ratio = float(np.mean(y_clean[sample_idx] != y_noisy[sample_idx]))
        real_noise_level[cid] = noise_ratio

        print(
            f"Client {cid}: gamma_s=1 target={noise_p:.4f} "
            f"k_target={k_target}/{n} real={noise_ratio:.4f} flipped={len(flipped_global)}/{n}"
        )

        # per-client per-class summary (your format)
        rows = per_class_flip_summary(y_clean, y_noisy, num_classes, sample_idx=sample_idx)
        print(f"  Per-class label noise summary (client {cid}):")
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

    return y_noisy, gamma_s, real_noise_level, flip_mask, flips_by_client
