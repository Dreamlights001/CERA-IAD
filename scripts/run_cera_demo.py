from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cera_iad.adapters.mock import make_synthetic_proposals
from cera_iad.core.conformal import ConformalGate
from cera_iad.core.memory import NormalMemory
from cera_iad.core.pipeline import CeraIAD
from cera_iad.core.scoring import ScoreWeights


def build_demo(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    dim = 32
    normal_center = rng.normal(0, 0.2, size=dim)
    memory_features = normal_center + rng.normal(0, 0.08, size=(128, dim))
    memory = NormalMemory(memory_features)

    calibration_scores = []
    for _ in range(40):
        proposals = make_synthetic_proposals(rng, normal_center, count=3, anomaly_shift=0.02)
        detector = CeraIAD(
            memory=memory,
            gate=ConformalGate(threshold=1.0),
            weights=ScoreWeights(),
        )
        calibration_scores.append(detector.evaluate_image("calib", proposals).image_score)

    gate = ConformalGate.fit(calibration_scores, epsilon=0.05, abstention_margin=0.04)
    detector = CeraIAD(memory=memory, gate=gate, weights=ScoreWeights())
    proposals = make_synthetic_proposals(rng, normal_center, count=4, anomaly_shift=1.2)
    output = detector.evaluate_image("demo-image-0001", proposals)
    return output.to_json_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic CERA-IAD smoke demo.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/cera_demo_evidence.json"))
    args = parser.parse_args()

    result = build_demo(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
