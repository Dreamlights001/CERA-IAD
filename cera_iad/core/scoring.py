from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cera_iad.core.types import RegionProposal, RetrievalEvidence


@dataclass(frozen=True)
class ScoreWeights:
    semantic: float = 0.30
    rarity: float = 0.40
    disagreement: float = 0.20
    uncertainty: float = 0.10

    def normalized(self) -> "ScoreWeights":
        total = self.semantic + self.rarity + self.disagreement + self.uncertainty
        if total <= 0:
            raise ValueError("score weights must sum to a positive value")
        return ScoreWeights(
            semantic=self.semantic / total,
            rarity=self.rarity / total,
            disagreement=self.disagreement / total,
            uncertainty=self.uncertainty / total,
        )


@dataclass(frozen=True)
class ScoreComponents:
    semantic: float
    rarity: float
    disagreement: float
    uncertainty: float
    fused: float


def fuse_scores(
    proposal: RegionProposal,
    retrieval: RetrievalEvidence,
    weights: ScoreWeights,
) -> ScoreComponents:
    weights = weights.normalized()
    semantic = _nonnegative(proposal.semantic_score)
    rarity = _nonnegative(retrieval.rarity_score)
    retrieval_uncertainty = retrieval.neighbor_entropy * _squash(retrieval.rarity_score)
    uncertainty = _nonnegative(max(proposal.uncertainty_score, retrieval_uncertainty))
    disagreement = evidence_disagreement(semantic, rarity, proposal.mask)
    fused = (
        weights.semantic * semantic
        + weights.rarity * rarity
        + weights.disagreement * disagreement
        + weights.uncertainty * uncertainty
    )
    return ScoreComponents(
        semantic=float(semantic),
        rarity=float(rarity),
        disagreement=float(disagreement),
        uncertainty=float(uncertainty),
        fused=float(fused),
    )


def evidence_disagreement(semantic: float, rarity: float, mask: np.ndarray | None) -> float:
    score_gap = abs(float(semantic) - float(rarity))
    if mask is None:
        return score_gap
    mask = np.asarray(mask)
    if mask.size == 0:
        return score_gap
    mask_density = float(np.mean(mask > 0))
    geometry_penalty = abs(mask_density - 0.15)
    return score_gap + 0.25 * geometry_penalty


def _nonnegative(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(float(value), 0.0)


def _squash(value: float) -> float:
    value = _nonnegative(value)
    return value / (1.0 + value)
