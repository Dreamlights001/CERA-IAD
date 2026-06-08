from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cera_iad.adapters.proposals import proposals_from_dicts
from cera_iad.core.types import ImageEvidence, RegionProposal


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_evidence_json(path: str | Path, evidence: ImageEvidence) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence.to_json_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_proposals(path: str | Path) -> list[RegionProposal]:
    """Load proposals from a JSON file.

    Expected schema:
    [
      {
        "region_id": "r00",
        "bbox_xyxy": [x1, y1, x2, y2],
        "patch_feature": [float, ...],
        "semantic_score": 0.2,
        "uncertainty_score": 0.1,
        "source": "AA-CLIP"
      }
    ]
    """

    items = load_json(path)
    if not isinstance(items, list):
        raise ValueError("proposal JSON must be a list of region proposal objects")
    return proposals_from_dicts(items)


def load_scores(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    values = load_json(path)
    return np.asarray(values, dtype=np.float64)
