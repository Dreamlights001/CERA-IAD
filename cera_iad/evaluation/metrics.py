from __future__ import annotations

import numpy as np


def expected_calibration_error(
    confidences: list[float] | np.ndarray,
    correctness: list[int] | np.ndarray,
    bins: int = 10,
) -> float:
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)
    if conf.shape != corr.shape:
        raise ValueError("confidences and correctness must have the same shape")
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not np.any(mask):
            continue
        acc = float(corr[mask].mean())
        avg_conf = float(conf[mask].mean())
        ece += float(mask.mean()) * abs(acc - avg_conf)
    return float(ece)


def abstention_gain(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    decisions: list[str],
) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or len(labels) != len(decisions):
        raise ValueError("labels, scores, and decisions must have equal length")
    predicted = (scores > 0.5).astype(int)
    base_error = float(np.mean(predicted != labels)) if len(labels) else 0.0
    keep = np.asarray([d != "review" for d in decisions], dtype=bool)
    if not np.any(keep):
        return base_error
    kept_error = float(np.mean(predicted[keep] != labels[keep]))
    return float(base_error - kept_error)


def retrieval_purity(neighbor_labels: list[int] | np.ndarray) -> float:
    labels = np.asarray(neighbor_labels, dtype=int)
    if len(labels) == 0:
        return 0.0
    return float(np.mean(labels == 0))
