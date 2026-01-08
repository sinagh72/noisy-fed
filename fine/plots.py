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
    Returns per-class total flips and flip rates.
    """
    y_clean = np.asarray(y_clean).astype(int)
    y_noisy = np.asarray(y_noisy).astype(int)

    rows = []
    for k in range(num_classes):
        mask = (y_clean == k)
        total = int(mask.sum())
        flipped = int(np.sum(y_noisy[mask] != y_clean[mask]))
        kept = total - flipped
        flip_rate = flipped / total if total > 0 else 0.0

        rows.append({
            "class": k,
            "total": total,
            "flipped": flipped,
            "kept": kept,
            "flip_rate": flip_rate,
        })

    return rows