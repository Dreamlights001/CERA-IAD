from __future__ import annotations

import numpy as np

from cera_iad.core.types import RegionProposal


def make_synthetic_proposals(
    rng: np.random.Generator,
    normal_center: np.ndarray,
    count: int = 4,
    anomaly_shift: float = 1.2,
) -> list[RegionProposal]:
    """Generate deterministic toy proposals for smoke tests and demos."""

    proposals: list[RegionProposal] = []
    dim = len(normal_center)
    for i in range(count):
        is_last = i == count - 1
        shift = anomaly_shift if is_last else 0.05
        feature = normal_center + rng.normal(shift, 0.08, size=dim)
        semantic = 0.75 if is_last else 0.12 + 0.05 * i
        uncertainty = 0.25 if is_last else 0.08
        proposals.append(
            RegionProposal(
                region_id=f"r{i:02d}",
                bbox_xyxy=(8 + 12 * i, 8, 24 + 12 * i, 24),
                patch_feature=feature.astype(np.float32),
                semantic_score=semantic,
                uncertainty_score=uncertainty,
                source="mock-aa-clip",
                metadata={"synthetic_anomaly": is_last},
            )
        )
    return proposals
