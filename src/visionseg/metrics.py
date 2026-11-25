"""Confusion counts, rank metrics and image-level uncertainty."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion(prediction, target):
    p, y = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if p.shape != y.shape:
        raise ValueError("Prediction/target shapes differ")
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    fn, tn = int((~p & y).sum()), int((~p & ~y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.,
            "accuracy": (tp + tn) / max(1, p.size)}


def ranking(labels, scores):
    y, s = np.asarray(labels).ravel(), np.asarray(scores).ravel()
    if y.shape != s.shape or not np.isfinite(s).all():
        raise ValueError("Invalid labels/scores")
    if np.unique(y).size < 2:
        return {"auroc": None, "average_precision": None}
    return {"auroc": float(roc_auc_score(y, s)),
            "average_precision": float(average_precision_score(y, s))}


def wilson_interval(successes, total, z=1.95996398454):
    if not total:
        return [None, None]
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [0. if successes == 0 else max(0., float(center - half)),
            1. if successes == total else min(1., float(center + half))]


def bootstrap_auroc(labels, scores, seed=0, repeats=200):
    """Stratified bootstrap over images, never over correlated pixels."""
    y, s = np.asarray(labels), np.asarray(scores)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        return [None, None]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(positive, len(positive)), rng.choice(negative, len(negative))])
        values.append(roc_auc_score(y[idx], s[idx]))
    return np.quantile(values, [.025, .975]).tolist()
