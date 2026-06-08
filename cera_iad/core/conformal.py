from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalGate:
    """Conformal threshold with an abstention band around the decision boundary."""

    threshold: float
    epsilon: float = 0.05
    abstention_margin: float = 0.05

    @classmethod
    def fit(
        cls,
        normal_scores: list[float] | np.ndarray,
        epsilon: float = 0.05,
        abstention_margin: float = 0.05,
    ) -> "ConformalGate":
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")
        scores = np.asarray(normal_scores, dtype=np.float64)
        if scores.ndim != 1 or len(scores) == 0:
            raise ValueError("normal_scores must be a non-empty 1D array")
        scores = np.sort(scores)
        n = len(scores)
        rank = int(np.ceil((n + 1) * (1 - epsilon))) - 1
        rank = min(max(rank, 0), n - 1)
        return cls(
            threshold=float(scores[rank]),
            epsilon=float(epsilon),
            abstention_margin=float(abstention_margin),
        )

    def decide(self, score: float) -> str:
        low, high = self.abstention_band
        if score >= high:
            return "anomaly"
        if score <= low:
            return "normal"
        return "review"

    @property
    def abstention_band(self) -> tuple[float, float]:
        return (
            float(self.threshold - self.abstention_margin),
            float(self.threshold + self.abstention_margin),
        )
