import numpy as np
import os

def per_class_metrics(y_true, y_pred, num_classes, eps=1e-12):
    """
    Multi-class, per-class metrics using one-vs-rest:
      TP_c = #(y_true=c and y_pred=c)
      FP_c = #(y_true!=c and y_pred=c)
      FN_c = #(y_true=c and y_pred!=c)
      TN_c = #(y_true!=c and y_pred!=c)

    Returns:
      metrics: dict with arrays of length C for TP/FP/FN/TN, precision, recall, f1, support
      plus macro/micro summaries.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    C = int(num_classes)
    conf = np.zeros((C, C), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < C) & (y_pred >= 0) & (y_pred < C)
    yt = y_true[valid]
    yp = y_pred[valid]

    # confusion matrix: rows=true, cols=pred
    np.add.at(conf, (yt, yp), 1)

    TP = np.diag(conf).astype(np.int64)
    FP = conf.sum(axis=0) - TP
    FN = conf.sum(axis=1) - TP
    TN = conf.sum() - (TP + FP + FN)

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    support   = conf.sum(axis=1)

    # overall accuracy
    acc = TP.sum() / (conf.sum() + eps)

    # macro averages
    macro_precision = precision.mean()
    macro_recall    = recall.mean()
    macro_f1        = f1.mean()

    # micro averages (for single-label multiclass, micro P=R=F1=accuracy)
    micro_TP = TP.sum()
    micro_FP = FP.sum()
    micro_FN = FN.sum()
    micro_precision = micro_TP / (micro_TP + micro_FP + eps)
    micro_recall    = micro_TP / (micro_TP + micro_FN + eps)
    micro_f1        = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)

    return {
        "confusion": conf,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision, "recall": recall, "f1": f1, "support": support,
        "acc": acc,
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        
    }



def save_metrics_report_txt(path, title, metrics_dict, num_classes, class_names=None, extra_lines=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(title + "\n")
        f.write("=" * 80 + "\n")

        if extra_lines:
            for line in extra_lines:
                f.write(str(line) + "\n")
            f.write("-" * 80 + "\n")

        f.write(f"Overall acc:  {metrics_dict['acc']:.6f}\n")
        f.write(f"Macro  F1:    {metrics_dict['macro']['f1']:.6f}\n")
        f.write(f"Micro  F1:    {metrics_dict['micro']['f1']:.6f}\n")
        f.write("\nPer-class:\n")
        f.write("class\tname\tprecision\trecall\tf1\tsupport\tTP\tFP\tFN\tTN\n")

        for c in range(num_classes):
            name = class_names[c] if class_names is not None else str(c)
            f.write(
                f"{c}\t{name}\t"
                f"{metrics_dict['precision'][c]:.6f}\t{metrics_dict['recall'][c]:.6f}\t{metrics_dict['f1'][c]:.6f}\t"
                f"{metrics_dict['support'][c]}\t{metrics_dict['TP'][c]}\t{metrics_dict['FP'][c]}\t{metrics_dict['FN'][c]}\t{metrics_dict['TN'][c]}\n"
            )
