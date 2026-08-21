"""Accuracy/confusion-matrix reporting, shared by both models."""

import numpy as np


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def print_classification_report(preds: np.ndarray, labels: np.ndarray, class_names: list) -> None:
    cm = confusion_matrix(preds, labels, len(class_names))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    for idx, name in enumerate(class_names):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        print(f"  {name}: precision={precision:.3f} recall={recall:.3f}")


def balanced_accuracy(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Macro-average of per-class recall -- every class counts equally
    regardless of how many test examples it has, unlike plain accuracy
    (correct / total), which is dominated by whichever classes have the
    most examples. The metric to watch when classes are imbalanced (as
    this project's cocoimages/ dataset is, by roughly 9x between the
    smallest and largest class) -- a model can post high plain accuracy
    while quietly failing the smaller classes, and this is what catches
    that. Classes with zero test examples are excluded from the average
    (not counted as 0), rather than skewing it for an unrelated reason."""
    cm = confusion_matrix(preds, labels, num_classes)
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class_recall = np.diag(cm) / support
    return float(np.nanmean(np.where(support > 0, per_class_recall, np.nan)))


def compute_precision_recall(preds: np.ndarray, labels: np.ndarray, class_names: list) -> dict:
    cm = confusion_matrix(preds, labels, len(class_names))
    result = {}
    for idx, name in enumerate(class_names):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        result[name] = {
            "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
            "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        }
    return result


def print_accuracy_table(accuracy_table: dict) -> None:
    """accuracy_table: {split: {column: value}}, e.g.
    {"train": {"float": 0.51, "qat": 0.53}, "val": {...}, "test": {...}}.
    Column set is inferred from the first split so MLP (float/qat/int8)
    and CNN (float only) both render correctly."""
    splits = [s for s in ("train", "val", "test") if s in accuracy_table]
    columns = list(next(iter(accuracy_table.values())).keys())

    header = f"{'split':<6} | " + " | ".join(f"{c:>8}" for c in columns)
    print(header)
    print("-" * len(header))
    for split_name in splits:
        row = accuracy_table[split_name]
        print(f"{split_name:<6} | " + " | ".join(f"{row[c]:>8.4f}" for c in columns))
