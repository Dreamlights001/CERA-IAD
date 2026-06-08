from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cera_iad.core.types import Array, RetrievalEvidence


def l2_normalize(features: Array, eps: float = 1e-12) -> Array:
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norms, eps)


@dataclass
class NormalMemory:
    """Normal image/patch feature memory with simple exact kNN retrieval."""

    features: Array
    ids: list[str] | None = None
    normalize: bool = True

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"memory features must be 2D, got shape {features.shape}")
        if len(features) == 0:
            raise ValueError("memory features cannot be empty")
        if self.normalize:
            features = l2_normalize(features)
        self.features = features
        if self.ids is not None and len(self.ids) != len(self.features):
            raise ValueError("ids length must match feature count")

    @classmethod
    def from_npy(cls, path: str, normalize: bool = True) -> "NormalMemory":
        return cls(np.load(path), normalize=normalize)

    def query(self, feature: Array, k: int = 5) -> RetrievalEvidence:
        if k <= 0:
            raise ValueError("k must be positive")
        feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        if self.normalize:
            feature = l2_normalize(feature)
        if feature.shape[1] != self.features.shape[1]:
            raise ValueError(
                f"query dim {feature.shape[1]} does not match memory dim {self.features.shape[1]}"
            )
        distances = np.linalg.norm(self.features - feature, axis=1)
        k_eff = min(k, len(distances))
        nn = np.argpartition(distances, k_eff - 1)[:k_eff]
        nn = nn[np.argsort(distances[nn])]
        nn_distances = distances[nn].astype(float)
        rarity = float(nn_distances.mean())
        entropy = _distance_entropy(nn_distances)
        return RetrievalEvidence(
            neighbor_indices=[int(i) for i in nn],
            neighbor_distances=[float(d) for d in nn_distances],
            rarity_score=rarity,
            neighbor_entropy=entropy,
        )


def _distance_entropy(distances: Array, eps: float = 1e-12) -> float:
    if len(distances) <= 1:
        return 0.0
    sim = np.exp(-np.asarray(distances, dtype=np.float64))
    prob = sim / max(float(sim.sum()), eps)
    entropy = float(-(prob * np.log(prob + eps)).sum())
    return entropy / float(np.log(len(distances)))
