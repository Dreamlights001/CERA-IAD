from __future__ import annotations

import argparse
from pathlib import Path

from cera_iad.core.conformal import ConformalGate
from cera_iad.core.io import load_proposals, load_scores, save_evidence_json
from cera_iad.core.memory import NormalMemory
from cera_iad.core.pipeline import CeraIAD
from cera_iad.core.scoring import ScoreWeights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CERA-IAD from pre-extracted memory features and proposal JSON."
    )
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--memory-npy", required=True, help="2D normal patch/image feature memory.")
    parser.add_argument("--proposals-json", required=True, help="Region proposals with patch features.")
    parser.add_argument(
        "--calibration-scores",
        required=True,
        help="Normal calibration scores, as .npy or JSON list.",
    )
    parser.add_argument("--output", default="outputs/cera_evidence.json")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--abstention-margin", type=float, default=0.05)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--semantic-weight", type=float, default=0.30)
    parser.add_argument("--rarity-weight", type=float, default=0.40)
    parser.add_argument("--disagreement-weight", type=float, default=0.20)
    parser.add_argument("--uncertainty-weight", type=float, default=0.10)
    args = parser.parse_args()

    memory = NormalMemory.from_npy(args.memory_npy)
    proposals = load_proposals(args.proposals_json)
    calibration_scores = load_scores(args.calibration_scores)
    gate = ConformalGate.fit(
        calibration_scores,
        epsilon=args.epsilon,
        abstention_margin=args.abstention_margin,
    )
    detector = CeraIAD(
        memory=memory,
        gate=gate,
        weights=ScoreWeights(
            semantic=args.semantic_weight,
            rarity=args.rarity_weight,
            disagreement=args.disagreement_weight,
            uncertainty=args.uncertainty_weight,
        ),
        retrieval_k=args.retrieval_k,
    )
    evidence = detector.evaluate_image(args.image_id, proposals)
    save_evidence_json(Path(args.output), evidence)
    print(f"Saved evidence JSON to {args.output}")
    print(f"decision={evidence.decision} score={evidence.image_score:.4f}")


if __name__ == "__main__":
    main()
