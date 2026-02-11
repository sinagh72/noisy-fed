import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
import os
import hashlib


def make_unique_labels(folders):
    # First attempt: basename
    base = [os.path.basename(os.path.normpath(p)) for p in folders]

    # If unique, done
    if len(set(base)) == len(base):
        return base

    # Second attempt: last 4 path components (usually enough context)
    pretty = []
    for p in folders:
        parts = os.path.normpath(p).split(os.sep)
        pretty.append("/".join(parts[-4:]))  # e.g., "asym/5 clean/0.3/fedavg"

    # If unique now, done
    if len(set(pretty)) == len(pretty):
        return pretty

    # Final fallback: force uniqueness by suffixing
    counts = defaultdict(int)
    unique = []
    for lab in pretty:
        counts[lab] += 1
        unique.append(lab if counts[lab] == 1 else f"{lab}__{counts[lab]}")
    return unique


def process_results(filepath, max_rounds=None):
    """
    Parse logs like:

    ================ Round 92 ================
    Overall acc:  0.866900
    Macro  F1:    0.866405
    Micro  F1:    0.866900
    Loss:         0.455554

    Returns:
      epochs: list[int]  (round numbers)
      results: dict[str, list[float]] keys: "accuracy", "f1", "loss"
    """
    try:
        with open(filepath, "r") as f:
            data = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

    # Capture round blocks
    round_pat = re.compile(r"^=+\s*Round\s+(\d+)\s*=+\s*$", flags=re.MULTILINE)
    matches = list(round_pat.finditer(data))
    if not matches:
        # fallback: sometimes '=' count differs but "Round XX" line exists
        round_pat = re.compile(r"^\s*=+\s*Round\s+(\d+)\s*=+.*$", flags=re.MULTILINE)
        matches = list(round_pat.finditer(data))

    if not matches:
        print(f"[WARN] No rounds found in: {filepath}")
        return None, None

    blocks = []
    for i, m in enumerate(matches):
        rnum = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(data)
        blocks.append((rnum, data[start:end]))

    # Extract metrics
    epochs = []
    results = defaultdict(list)

    acc_pat  = re.compile(r"Overall\s+acc:\s*([0-9]*\.?[0-9]+)")
    f1_pat   = re.compile(r"Macro\s+F1:\s*([0-9]*\.?[0-9]+)")
    loss_pat = re.compile(r"Loss:\s*([0-9]*\.?[0-9]+)")

    for rnum, block in blocks:
        acc_m = acc_pat.search(block)
        f1_m = f1_pat.search(block)
        loss_m = loss_pat.search(block)

        # strict: require all three
        if not acc_m or not f1_m or not loss_m:
            continue

        epochs.append(rnum)
        results["accuracy"].append(float(acc_m.group(1)))
        results["f1"].append(float(f1_m.group(1)))
        results["loss"].append(float(loss_m.group(1)))

        if max_rounds is not None and len(epochs) >= max_rounds:
            break

    if not epochs:
        print(f"[WARN] Rounds found but no (acc,f1,loss) extracted in: {filepath}")
        return None, None

    return epochs, results


if __name__ == "__main__":
    # === configure methods ===
    folders = [
        "../results/asym_scratch/10 clean/0.0/fedavg",
        "../results/asym_scratch/5 clean/0.3/fedavg",
        "../results/asym_scratch/5 clean/0.3/pruning",
        # "../results/asym_scratch/5 clean/0.3/fedeig",
        "../results/asym/5 clean/0.3/fedeig",
        # "../results/oracle/scratch",
    ]
    labels = make_unique_labels(folders)
    colors = {label: f"C{i}" for i, label in enumerate(labels)}

    PLOT_METRICS = ["accuracy", "f1", "loss"]
    MAX_ROUNDS = 20
    MAX_LEN = 100

    # which metric you care most about for "top-counts"
    METRIC_FOR_SUMMARY = "accuracy"  # or "f1" or "loss"
    top_counts = defaultdict(int)

    # -------------------------------------------------------
    # 1. Discover all .txt log files across all method folders
    # -------------------------------------------------------
    files_per_label = {label: set() for label in labels}

    for folder, label in zip(folders, labels):
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".txt"):
                files_per_label[label].add(fname)

    all_filenames = sorted(set().union(*files_per_label.values()))
    print("Found log files:", all_filenames)

    # output folder name based on method labels
    out_folder = "_".join(labels)
    if len(out_folder) > MAX_LEN:
        digest = hashlib.md5(out_folder.encode("utf-8")).hexdigest()[:8]
        out_folder = f"plots_{digest}"
    os.makedirs(out_folder, exist_ok=True)

    # -------------------------------------------------------
    # 2. For each filename, load results from each method and plot
    # -------------------------------------------------------
    for filename in all_filenames:
        results_by_method = {}
        epochs_by_method = {}

        for folder, label in zip(folders, labels):
            log_path = os.path.join(folder, filename)
            if not os.path.exists(log_path):
                continue

            epochs, results = process_results(log_path, max_rounds=MAX_ROUNDS)
            if epochs is None:
                continue

            results_by_method[label] = results
            epochs_by_method[label] = epochs

        if not results_by_method:
            print(f"[INFO] No valid data for {filename}, skipping.")
            continue

        for metric in PLOT_METRICS:
            plt.figure(figsize=(12, 8))
            handles = []

            is_loss = (metric.lower() == "loss")
            best_label = None
            best_val = np.inf if is_loss else -np.inf
            best_round = None

            for label in results_by_method:
                res = results_by_method[label]
                if metric not in res:
                    continue

                x = np.array(epochs_by_method[label], dtype=np.int64)
                y = np.array(res[metric], dtype=np.float64)
                if y.size == 0:
                    continue

                # simple visual band (not std; just a tiny margin)
                margin = 0.1 * (y.max() - y.min()) if y.max() != y.min() else 0.001
                plt.fill_between(x, y - margin, y + margin, color=colors[label], alpha=0.25)
                plt.plot(x, y, color=colors[label], linewidth=2)

                # mark best point (min for loss, max otherwise)
                idx = int(np.argmin(y) if is_loss else np.argmax(y))
                extreme_val = float(y[idx])
                extreme_round = int(x[idx])

                plt.plot(
                    extreme_round,
                    extreme_val,
                    marker="x",
                    color=colors[label],
                    markersize=7,
                    markeredgewidth=2,
                )

                handles.append(Line2D([0], [0], color=colors[label], lw=2, label=label))
                tag = "Min" if is_loss else "Max"
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="x",
                        color=colors[label],
                        markersize=7,
                        markeredgewidth=2,
                        linestyle="None",
                        label=f"{label} {tag}: {extreme_val:.4f} (round {extreme_round})",
                    )
                )

                # update best across methods
                if (is_loss and extreme_val < best_val) or ((not is_loss) and extreme_val > best_val):
                    best_val = extreme_val
                    best_label = label
                    best_round = extreme_round

            if best_label is not None:
                direction = "min" if is_loss else "max"
                print(
                    f"Best ({direction}) approach for {metric}_{filename}: "
                    f"{best_label} ({best_val:.4f} at round {best_round})"
                )

                # count wins for chosen summary metric, using correct direction
                if metric == METRIC_FOR_SUMMARY:
                    top_counts[best_label] += 1

            plt.xlabel("Round", fontsize=16)
            plt.ylabel(metric, fontsize=16)
            plt.title(f"{filename} - {metric} Comparison (TEST net_final)", fontsize=18)
            plt.grid(True)

            # legend placement: loss often better upper-right, acc/f1 lower-right
            loc = "upper right" if is_loss else "lower right"
            plt.legend(handles=handles, fontsize=12, loc=loc)

            plt.tight_layout()

            save_fn = f"{filename}_{metric}_comparison.png"
            plt.savefig(os.path.join(out_folder, save_fn))
            plt.close()

    # === Summary ===
    print(f"\nSummary of top-counts for {METRIC_FOR_SUMMARY}:")
    for label, count in top_counts.items():
        print(f"  {label}: {count}")

    if top_counts:
        overall_best = max(top_counts, key=top_counts.get)
        print(
            f"\nApproach with most top-{METRIC_FOR_SUMMARY} wins: "
            f"{overall_best} ({top_counts[overall_best]} times)"
        )
