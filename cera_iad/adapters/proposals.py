from __future__ import annotations

from typing import Any

import numpy as np

from cera_iad.core.types import RegionProposal


def proposals_from_dicts(items: list[dict[str, Any]]) -> list[RegionProposal]:
    proposals: list[RegionProposal] = []
    for item in items:
        proposals.append(
            RegionProposal(
                region_id=str(item["region_id"]),
                bbox_xyxy=tuple(int(v) for v in item["bbox_xyxy"]),
                patch_feature=np.asarray(item["patch_feature"], dtype=np.float32),
                semantic_score=float(item.get("semantic_score", 0.0)),
                uncertainty_score=float(item.get("uncertainty_score", 0.0)),
                mask=np.asarray(item["mask"]) if item.get("mask") is not None else None,
                source=str(item.get("source", "external")),
                metadata=dict(item.get("metadata", {})),
            )
        )
    return proposals
