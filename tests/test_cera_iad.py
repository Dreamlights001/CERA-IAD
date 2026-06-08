from __future__ import annotations

import numpy as np

from cera_iad.adapters.mock import make_synthetic_proposals
from cera_iad.core.conformal import ConformalGate
from cera_iad.core.memory import NormalMemory
from cera_iad.core.pipeline import CeraIAD
from cera_iad.evaluation.metrics import abstention_gain, expected_calibration_error
from cera_iad.modules.registry import build_cloud_plan, registry_snapshot


def test_conformal_gate_orders_decisions() -> None:
    gate = ConformalGate.fit([0.1, 0.2, 0.3, 0.4], epsilon=0.25, abstention_margin=0.05)
    assert gate.decide(gate.threshold + 0.10) == "anomaly"
    assert gate.decide(gate.threshold - 0.10) == "normal"
    assert gate.decide(gate.threshold) == "review"


def test_pipeline_flags_synthetic_shifted_region() -> None:
    rng = np.random.default_rng(3)
    center = np.zeros(16, dtype=np.float32)
    memory = NormalMemory(rng.normal(0, 0.04, size=(64, 16)))
    gate = ConformalGate(threshold=0.35, abstention_margin=0.04)
    detector = CeraIAD(memory=memory, gate=gate, retrieval_k=3)
    proposals = make_synthetic_proposals(rng, center, count=4, anomaly_shift=1.0)
    evidence = detector.evaluate_image("x", proposals)
    assert evidence.decision == "anomaly"
    assert evidence.regions[-1].decision == "anomaly"
    assert evidence.regions[-1].retrieval.neighbor_indices
    assert evidence.regions[-1].fused_score > evidence.regions[0].fused_score


def test_metrics_are_bounded() -> None:
    ece = expected_calibration_error([0.1, 0.9, 0.8], [0, 1, 1], bins=3)
    gain = abstention_gain([0, 1, 1], [0.2, 0.8, 0.3], ["normal", "anomaly", "review"])
    assert 0.0 <= ece <= 1.0
    assert gain >= 0.0


def test_cloud_plan_resolves_ablation_modules_without_heavy_imports() -> None:
    config = {
        "method": "CERA-IAD",
        "modules": {
            "encoder": "clip_openai",
            "candidate_generator": "aa_clip",
            "memory_backend": "exact_knn",
            "mask_refiner": "no_mask",
            "reasoner": "no_reasoner",
            "calibrator": "conformal_global",
            "fusion_scorer": "linear_fusion",
        },
    }
    ablation = {
        "name": "A5_mllm_reflection",
        "modules": {
            "mask_refiner": "sam_box",
            "reasoner": "qwen25_vl",
        },
    }
    plan = build_cloud_plan(config, ablation)
    assert plan["ablation"] == "A5_mllm_reflection"
    assert plan["modules"]["reasoner"]["name"] == "qwen25_vl"
    assert "qwen25_vl_7b" in plan["required_weights"]
    assert "sam_vit_h" in plan["required_weights"]


def test_registry_contains_required_roles() -> None:
    snapshot = registry_snapshot()
    kinds = {item["kind"] for item in snapshot.values()}
    assert {
        "encoder",
        "candidate_generator",
        "memory_backend",
        "mask_refiner",
        "reasoner",
        "calibrator",
        "fusion_scorer",
    }.issubset(kinds)
