from __future__ import annotations

from dataclasses import dataclass, field

from cera_iad.core.conformal import ConformalGate
from cera_iad.core.memory import NormalMemory
from cera_iad.core.scoring import ScoreWeights, fuse_scores
from cera_iad.core.types import ImageEvidence, RegionEvidence, RegionProposal


@dataclass
class CeraIAD:
    """Evidence-calibrated anomaly detector over externally generated proposals."""

    memory: NormalMemory
    gate: ConformalGate
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    retrieval_k: int = 5
    reflection_disagreement: float = 0.35
    reflection_uncertainty: float = 0.50

    def evaluate_regions(self, proposals: list[RegionProposal]) -> list[RegionEvidence]:
        evidence: list[RegionEvidence] = []
        for proposal in proposals:
            retrieval = self.memory.query(proposal.patch_feature, k=self.retrieval_k)
            scores = fuse_scores(proposal, retrieval, self.weights)
            decision = self.gate.decide(scores.fused)
            needs_reflection = (
                decision == "review"
                or scores.disagreement >= self.reflection_disagreement
                or scores.uncertainty >= self.reflection_uncertainty
            )
            evidence.append(
                RegionEvidence(
                    region_id=proposal.region_id,
                    bbox_xyxy=proposal.bbox_xyxy,
                    semantic_score=scores.semantic,
                    rarity_score=scores.rarity,
                    disagreement_score=scores.disagreement,
                    uncertainty_score=scores.uncertainty,
                    fused_score=scores.fused,
                    decision=decision,
                    needs_reflection=needs_reflection,
                    retrieval=retrieval,
                    rationale=_rationale(scores.semantic, scores.rarity, scores.disagreement),
                    source=proposal.source,
                    metadata=proposal.metadata,
                )
            )
        return evidence

    def evaluate_image(self, image_id: str, proposals: list[RegionProposal]) -> ImageEvidence:
        regions = self.evaluate_regions(proposals)
        image_score = max((region.fused_score for region in regions), default=0.0)
        if not regions:
            decision = "review"
        elif any(region.decision == "anomaly" for region in regions):
            decision = "anomaly"
        elif any(region.decision == "review" for region in regions):
            decision = "review"
        else:
            decision = "normal"
        return ImageEvidence(
            image_id=image_id,
            image_score=float(image_score),
            decision=decision,
            conformal_threshold=self.gate.threshold,
            abstention_band=self.gate.abstention_band,
            regions=regions,
        )


def _rationale(semantic: float, rarity: float, disagreement: float) -> str:
    if semantic >= rarity and disagreement > 0.35:
        return "semantic prior is stronger than empirical rarity; verify possible VLM hallucination."
    if rarity > semantic and disagreement > 0.35:
        return "empirical rarity is stronger than semantic prior; verify possible subtle or unknown defect."
    if rarity > semantic:
        return "region is rare under normal memory and supported by local retrieval evidence."
    return "region is mainly supported by semantic anomaly prior."
