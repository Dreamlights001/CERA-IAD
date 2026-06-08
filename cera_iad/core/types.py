from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class RegionProposal:
    """A candidate anomalous region before evidence fusion."""

    region_id: str
    bbox_xyxy: tuple[int, int, int, int]
    patch_feature: Array
    semantic_score: float = 0.0
    uncertainty_score: float = 0.0
    mask: Array | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalEvidence:
    """Nearest normal neighbors used to estimate empirical rarity."""

    neighbor_indices: list[int]
    neighbor_distances: list[float]
    rarity_score: float
    neighbor_entropy: float


@dataclass(frozen=True)
class RegionEvidence:
    """Final evidence bundle for a proposal."""

    region_id: str
    bbox_xyxy: tuple[int, int, int, int]
    semantic_score: float
    rarity_score: float
    disagreement_score: float
    uncertainty_score: float
    fused_score: float
    decision: str
    needs_reflection: bool
    retrieval: RetrievalEvidence
    rationale: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "scores": {
                "semantic": self.semantic_score,
                "rarity": self.rarity_score,
                "disagreement": self.disagreement_score,
                "uncertainty": self.uncertainty_score,
                "fused": self.fused_score,
            },
            "decision": self.decision,
            "needs_reflection": self.needs_reflection,
            "retrieval": {
                "neighbor_indices": self.retrieval.neighbor_indices,
                "neighbor_distances": self.retrieval.neighbor_distances,
                "neighbor_entropy": self.retrieval.neighbor_entropy,
            },
            "rationale": self.rationale,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ImageEvidence:
    """Image-level CERA-IAD output."""

    image_id: str
    image_score: float
    decision: str
    conformal_threshold: float
    abstention_band: tuple[float, float]
    regions: list[RegionEvidence]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_score": self.image_score,
            "decision": self.decision,
            "conformal_threshold": self.conformal_threshold,
            "abstention_band": list(self.abstention_band),
            "regions": [region.to_json_dict() for region in self.regions],
        }
