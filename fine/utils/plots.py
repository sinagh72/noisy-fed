import numpy as np
import matplotlib.pyplot as plt
import os

def plot_transition_heatmap(
    M,
    save_path,
    normalize="row",
    dpi=200,
    title="Label transitions",
    show_numbers=True,
):
    """
    normalize:
      - None  : raw counts
      - "row" : row-normalized (each row sums to 1)
      - "all" : global normalization
    """
    M = np.asarray(M)
    A = M.astype(float)

    if normalize == "row":
        row_sums = A.sum(axis=1, keepdims=True)
        A = np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums > 0)
        suffix = " (row-normalized)"
        fmt = "{:.2f}"
    elif normalize == "all":
        s = A.sum()
        A = A / s if s > 0 else A
        suffix = " (global-normalized)"
        fmt = "{:.2e}"
    else:
        suffix = " (counts)"
        fmt = "{:d}"

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    im = ax.imshow(A, aspect="auto")

    ax.set_title(title + suffix)
    ax.set_xlabel("Noisy label (after)")
    ax.set_ylabel("Original label (before)")
    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_yticks(np.arange(M.shape[0]))

    # 🔑 Annotate cells
    if show_numbers:
        # Threshold for text color
        threshold = A.max() * 0.5 if A.max() > 0 else 0

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                val = A[i, j]
                if normalize is None:
                    text = fmt.format(int(M[i, j]))
                else:
                    text = fmt.format(val)

                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color="white" if val > threshold else "black",
                    fontsize=8,
                )

    # Optional: diagonal guide
    ax.plot(
        [-0.5, M.shape[1] - 0.5],
        [-0.5, M.shape[0] - 0.5],
        linestyle="--",
        linewidth=1,
        color="gray",
        alpha=0.6,
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


def label_transition_matrix(y_before, y_after, num_classes):
    y_before = np.asarray(y_before).astype(int)
    y_after  = np.asarray(y_after).astype(int)

    M = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(M, (y_before, y_after), 1)   # counts transitions
    return M


def plot_label_distributions(
    y_before,
    y_after,
    num_classes,
    save_path,
    title_prefix="Label distribution",
    dpi=200,
):
    """
    Saves a single figure:
      (top)   counts before
      (mid)   counts after
      (bottom) delta (after - before) with markers
    """
    y_before = np.asarray(y_before)
    y_after  = np.asarray(y_after)

    # Ensure all classes appear (even if count is 0)
    cls = np.arange(num_classes)
    before_counts = np.bincount(y_before, minlength=num_classes)
    after_counts  = np.bincount(y_after,  minlength=num_classes)
    delta = after_counts - before_counts

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

    # --- Top: before ---
    axes[0].bar(cls, before_counts)
    axes[0].set_title(f"{title_prefix} (Before noise)")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # --- Middle: after ---
    axes[1].bar(cls, after_counts)
    axes[1].set_title(f"{title_prefix} (After noise)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # --- Bottom: difference ---
    axes[2].axhline(0, linewidth=1)
    axes[2].bar(cls, delta, alpha=0.7)              # bars show +/- change
    axes[2].plot(cls, delta, marker="o", linestyle="")  # markers emphasize change
    axes[2].set_title("Difference (After − Before)")
    axes[2].set_ylabel("Δ Count")
    axes[2].set_xlabel("Class index")
    axes[2].set_xticks(cls)
    axes[2].set_xticklabels(cls.astype(int))
    axes[2].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def per_class_flip_summary(y_clean, y_noisy, num_classes):
    """
    Per-class label corruption summary.

    For each class k:
      - total_before        : # samples originally in class k
      - flipped_out         : # samples that left class k due to noise
      - flipped_in          : # samples that entered class k due to noise
      - kept                : # samples that stayed in class k
      - flip_rate           : flipped_out / total_before
      - total_after_flipped : # samples labeled as class k after noise
      - delta               : after - before
    """
    y_clean = np.asarray(y_clean).astype(int)
    y_noisy = np.asarray(y_noisy).astype(int)

    rows = []
    for k in range(num_classes):
        mask_before = (y_clean == k)
        total_before = int(mask_before.sum())

        flipped_out = int(np.sum(y_noisy[mask_before] != y_clean[mask_before]))
        kept = total_before - flipped_out
        flip_rate = flipped_out / total_before if total_before > 0 else 0.0

        total_after_flipped = int(np.sum(y_noisy == k))
        flipped_in = int(np.sum((y_noisy == k) & (y_clean != k)))
        delta = total_after_flipped - total_before

        rows.append({
            "class": k,
            "total_before": total_before,
            "flipped_out": flipped_out,
            "flipped_in": flipped_in,
            "kept": kept,
            "flip_rate": float(flip_rate),
            "total_after_flipped": total_after_flipped,
            "delta": delta,
        })

    return rows


def clean_id_metrics(clean_mask: np.ndarray, y_noisy: np.ndarray, y_clean: np.ndarray):
    """
    CLEAN is the positive class.

    clean_mask: True means predicted CLEAN, False means predicted NOISY
    true_clean: (y_noisy == y_clean)

    TP: predicted clean AND truly clean
    FP: predicted clean BUT truly noisy   (this is contamination of the clean set)
    FN: predicted noisy BUT truly clean   (clean samples discarded)
    TN: predicted noisy AND truly noisy
    """
    true_clean = (y_noisy == y_clean)
    pred_clean = clean_mask

    tp = np.sum(pred_clean & true_clean)
    fp = np.sum(pred_clean & (~true_clean))
    fn = np.sum((~pred_clean) & true_clean)
    tn = np.sum((~pred_clean) & (~true_clean))

    precision = tp / (tp + fp + 1e-12)   # purity of selected clean
    recall    = tp / (tp + fn + 1e-12)   # how many true-clean you kept
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    acc       = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }



def per_class_clean_stats(clean_mask: np.ndarray, y_noisy: np.ndarray, y_clean: np.ndarray, num_classes: int):
    """
    Per-class stats where CLEAN is positive.

    Grouping is done by noisy label (y_noisy == k), same as your current code.
    """
    true_clean = (y_noisy == y_clean)
    pred_clean = clean_mask

    rows = []
    for k in range(num_classes):
        m = (y_noisy == k)   # group by current/noisy label
        Nk = int(m.sum())
        if Nk == 0:
            rows.append({
                "class": k, "N": 0,
                "TP": 0, "FP": 0, "FN": 0, "TN": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
            })
            continue

        TP = int(np.sum(m & pred_clean & true_clean))
        FP = int(np.sum(m & pred_clean & (~true_clean)))   # contamination inside predicted clean
        FN = int(np.sum(m & (~pred_clean) & true_clean))   # clean discarded
        TN = int(np.sum(m & (~pred_clean) & (~true_clean)))

        prec = TP / (TP + FP + 1e-12)
        rec  = TP / (TP + FN + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)

        rows.append({
            "class": k, "N": Nk,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
        })

    return rows



def print_noise_metrics(title, cm, y_noisy, y_clean, num_classes):
    print(f"\n{title}")

    overall = clean_id_metrics(cm, y_noisy, y_clean)
    rows = per_class_clean_stats(cm, y_noisy, y_clean, num_classes)

    print(f"  Acc={overall['accuracy']:.4f} "
          f"Prec={overall['precision']:.4f} "
          f"Rec={overall['recall']:.4f} "
          f"F1={overall['f1']:.4f}")
    print(f"  TP={overall['tp']} FP={overall['fp']} "
          f"FN={overall['fn']} TN={overall['tn']}")

    print("  Per-class:")
    for r in rows:
        print(
            f"    cls={r['class']:2d} N={r['N']:5d} "
            f"TP={r['TP']:4d} FP={r['FP']:4d} "
            f"FN={r['FN']:4d} TN={r['TN']:4d} | "
            f"Prec={r['precision']:.3f} "
            f"Rec={r['recall']:.3f} "
            f"F1={r['f1']:.3f}"
        )


def correction_metrics(y_clean: np.ndarray, y_noisy: np.ndarray, y_corrected: np.ndarray, global_idx: np.ndarray,):
    """
    Summarize how a label-correction step affected the dataset.

    Args:
      y_clean:     [N] ground-truth clean labels (oracle)
      y_noisy:     [N] labels after synthetic noise injection (before correction)
      y_corrected: [N] labels after your correction step
      global_idx: indices you *flagged* as noisy (i.e., the set you attempted to correct)

    Returns:
      dict of global noise rate before/after and detailed outcomes on flagged samples.
    """
    y_clean = np.asarray(y_clean)
    y_noisy = np.asarray(y_noisy)
    y_corrected = np.asarray(y_corrected)

    global_idx = np.asarray(global_idx, dtype=np.int64)

    # mask of samples you flagged for correction
    flagged_mask = np.zeros_like(y_clean, dtype=bool)
    flagged_mask[global_idx] = True

    # ground truth: which samples were truly noisy BEFORE correction
    was_wrong_before = (y_noisy != y_clean)
    was_correct_before = ~was_wrong_before

    # global noise rates
    noise_rate_before = float(was_wrong_before.mean())
    noise_rate_after = float((y_corrected != y_clean).mean())

    # among flagged samples:
    flagged_truly_noisy = flagged_mask & was_wrong_before   # good flags
    flagged_truly_clean = flagged_mask & was_correct_before # false positives

    # outcomes for flagged truly-noisy samples
    fixed_wrong = flagged_truly_noisy & (y_corrected == y_clean)
    still_wrong = flagged_truly_noisy & (y_corrected != y_clean)

    # outcomes for flagged truly-clean samples
    harmed_clean = flagged_truly_clean & (y_corrected != y_clean)  # you broke them
    safe_clean = flagged_truly_clean & (y_corrected == y_clean)    # harmless FP

    # rates (avoid divide-by-zero)
    n_flagged_truly_noisy = max(int(flagged_truly_noisy.sum()), 1)
    n_flagged_truly_clean = max(int(flagged_truly_clean.sum()), 1)

    fix_rate = float(fixed_wrong.sum() / n_flagged_truly_noisy)
    harm_rate = float(harmed_clean.sum() / n_flagged_truly_clean)

    # bookkeeping: how many labels you actually changed inside the flagged set
    num_flagged = int(flagged_mask.sum())
    num_changed_within_flagged = int(np.sum(flagged_mask & (y_corrected != y_noisy)))

    return {
        # global
        "noise_rate_before": noise_rate_before,
        "noise_rate_after": noise_rate_after,

        # flagged set size + action
        "num_flagged": num_flagged,
        "num_changed_within_flagged": num_changed_within_flagged,

        # flagged truly noisy
        "flagged_truly_noisy": int(flagged_truly_noisy.sum()),
        "fixed_wrong": int(fixed_wrong.sum()),
        "still_wrong": int(still_wrong.sum()),
        "fix_rate": fix_rate,

        # flagged truly clean (false positives)
        "flagged_truly_clean": int(flagged_truly_clean.sum()),
        "harmed_clean": int(harmed_clean.sum()),
        "safe_clean": int(safe_clean.sum()),
        "harm_rate": harm_rate,
    }



import numpy as np
import matplotlib.pyplot as plt

def plot_score_distributions_by_class(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    class_names=None,
    bins: int = 50,
    log_y: bool = False,
    title_prefix: str = "Score distribution",
    save_prefix: str | None = None,
):
    """
    Plot per-class score distributions.
    - Histogram grid (one subplot per class)
    - Boxplot (all classes side-by-side)
    - Prints per-class summary stats

    scores: shape [N]
    labels: shape [N] (int class id for each score)
    class_names: list[str] of length K (optional)
    """
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    assert scores.shape[0] == labels.shape[0], "scores and labels must align"

    classes = np.unique(labels)
    K = len(classes)

    # ---------- Summary stats ----------
    print("\nPer-class score stats:")
    for c in classes:
        s = scores[labels == c]
        if s.size == 0:
            continue
        print(
            f"  class {c}: n={s.size:6d}  "
            f"mean={s.mean():.4f}  std={s.std(ddof=0):.4f}  "
            f"p5={np.percentile(s,5):.4f}  p50={np.percentile(s,50):.4f}  p95={np.percentile(s,95):.4f}"
        )

    # ---------- Histogram grid ----------
    # Make a near-square grid
    ncols = int(np.ceil(np.sqrt(K)))
    nrows = int(np.ceil(K / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
    axes = axes.ravel()

    for i, c in enumerate(classes):
        ax = axes[i]
        s = scores[labels == c]
        name = class_names[c] if (class_names is not None and c < len(class_names)) else str(c)
        ax.hist(s, bins=bins)  # default style/colors
        ax.set_title(f"Class {name} (n={s.size})")
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        if log_y:
            ax.set_yscale("log")

    # Turn off unused axes
    for j in range(K, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{title_prefix} (histograms per class)")
    fig.tight_layout()

    if save_prefix:
        fig.savefig(f"{save_prefix}_hist_by_class.png", dpi=200)

    # ---------- Boxplot (distribution comparison) ----------
    data = [scores[labels == c] for c in classes]
    tick_labels = [
        (class_names[c] if (class_names is not None and c < len(class_names)) else str(c))
        for c in classes
    ]

    fig2 = plt.figure(figsize=(max(8, 0.8*K), 4))
    plt.boxplot(data, labels=tick_labels, showfliers=True)
    plt.title(f"{title_prefix} (boxplot across classes)")
    plt.xlabel("class")
    plt.ylabel("score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig2.savefig(f"{save_prefix}_boxplot_by_class.png", dpi=200)




