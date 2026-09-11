from __future__ import annotations

import torch


def classification_metrics(
    predictions: list[int], targets: list[int], num_classes: int = 4
) -> dict[str, float | list[float]]:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    for predicted, target in zip(predictions, targets, strict=True):
        confusion[target, predicted] += 1
    true_positive = confusion.diag()
    precision = true_positive / confusion.sum(0).clamp_min(1)
    recall = true_positive / confusion.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    accuracy = true_positive.sum() / confusion.sum().clamp_min(1)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * confusion.sum(1) / confusion.sum().clamp_min(1)).sum()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
    }
