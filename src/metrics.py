"""Evaluation metrics reported in the paper.

10-class metrics: top-1 / top-3 accuracy, macro recall / precision, class-wise
recall / precision, and the confusion matrix.

Binary safe-vs-distracted metrics (class 0 = safe, classes 1..9 = distracted):
2-class AUPRC and 2-class FNR.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve


def compute_metrics(true_labels, pred_labels, logits, num_classes, class_names=None):
    """Compute all paper metrics and return a nested dict."""
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    logits = np.asarray(logits)
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))

    classwise = []
    recalls, precisions = [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recalls.append(recall)
        precisions.append(precision)
        classwise.append(
            {
                "class_idx": i,
                "class_name": class_names[i],
                "recall": float(recall),
                "precision": float(precision),
                "support": int(cm[i, :].sum()),
            }
        )

    top1 = float(np.mean(pred_labels == true_labels))
    top3 = _top_k_accuracy(true_labels, logits, k=3)
    macro_recall = float(np.mean(recalls))
    macro_precision = float(np.mean(precisions))

    auprc, fnr = _binary_metrics(true_labels, pred_labels, logits)

    return {
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "binary_auprc": auprc,
        "binary_fnr": fnr,
        "classwise": classwise,
        "confusion_matrix": cm.tolist(),
        "num_samples": int(len(true_labels)),
    }


def _top_k_accuracy(true_labels, logits, k=3):
    k = min(k, logits.shape[1])
    topk = np.argsort(logits, axis=1)[:, -k:]
    hits = np.any(topk == true_labels[:, None], axis=1)
    return float(np.mean(hits))


def _binary_metrics(true_labels, pred_labels, logits):
    """Safe (class 0) vs distracted (classes 1..9)."""
    y_true = (true_labels != 0).astype(int)
    if logits.shape[1] > 1:
        y_scores = logits[:, 1:].max(axis=1)
    else:
        y_scores = logits[:, 0]

    if len(np.unique(y_true)) < 2:
        auprc = float("nan")
    else:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
        auprc = float(auc(rec_curve, prec_curve))

    # FNR: fraction of distracted images predicted as safe.
    danger = true_labels != 0
    fn = int(np.sum(danger & (pred_labels == 0)))
    tp = int(np.sum(danger & (pred_labels != 0)))
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    return auprc, fnr


def format_summary(metrics: dict) -> str:
    """Return a short, human-readable metrics summary (percentages)."""
    return (
        f"Top-1: {metrics['top1_accuracy'] * 100:.1f}  "
        f"Top-3: {metrics['top3_accuracy'] * 100:.1f}  "
        f"Recall: {metrics['macro_recall'] * 100:.1f}  "
        f"Precision: {metrics['macro_precision'] * 100:.1f}  "
        f"2C-AUPRC: {metrics['binary_auprc'] * 100:.1f}  "
        f"2C-FNR: {metrics['binary_fnr'] * 100:.1f}"
    )
