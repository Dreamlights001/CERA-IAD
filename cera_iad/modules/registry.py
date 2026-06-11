from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModuleSpec:
    """Lightweight descriptor for a replaceable CERA-IAD module.

    The registry stores provenance and command templates only. Heavy model
    imports and inference are intentionally left to adapter implementations.
    """

    name: str
    kind: str
    source: str
    baseline_path: str | None = None
    weight_keys: list[str] = field(default_factory=list)
    command_hint: str | None = None
    output_contract: str = "RegionProposal/ImageEvidence compatible JSON"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REGISTRY: dict[str, ModuleSpec] = {
    "clip_openai": ModuleSpec(
        name="clip_openai",
        kind="encoder",
        source="Hugging Face openai/clip-vit-large-patch14 or open_clip",
        weight_keys=["clip_vit_l_14"],
        command_hint="extract CLIP image/patch features to .npy",
        notes="Default semantic encoder for A0 and feature extraction.",
    ),
    "dinov2_large": ModuleSpec(
        name="dinov2_large",
        kind="encoder",
        source="Hugging Face facebook/dinov2-large",
        weight_keys=["dinov2_large"],
        command_hint="extract DINOv2 patch features to .npy",
        notes="Language-free encoder ablation.",
    ),
    "anomaly_vfm": ModuleSpec(
        name="anomaly_vfm",
        kind="encoder",
        source="baseline AnomalyVFM",
        baseline_path="../baselines/AnomalyVFM",
        weight_keys=["anomaly_vfm_optional"],
        notes="Adapter should call upstream scripts rather than reimplementing VFM logic.",
    ),
    "clip_only": ModuleSpec(
        name="clip_only",
        kind="candidate_generator",
        source="CLIP score map from shared encoder features",
        weight_keys=["clip_vit_l_14"],
        output_contract="RegionProposal JSON with semantic_score and patch_feature",
        notes="A0 sanity candidate generator.",
    ),
    "aa_clip": ModuleSpec(
        name="aa_clip",
        kind="candidate_generator",
        source="AA-CLIP baseline",
        baseline_path="../baselines/AA-CLIP",
        weight_keys=["clip_vit_l_14"],
        command_hint="run upstream AA-CLIP inference, then convert predictions to RegionProposal JSON",
    ),
    "anomaly_clip": ModuleSpec(
        name="anomaly_clip",
        kind="candidate_generator",
        source="AnomalyCLIP baseline",
        baseline_path="../baselines/AnomalyCLIP",
        weight_keys=["clip_vit_l_14"],
        command_hint="run upstream AnomalyCLIP inference and convert heatmaps to proposals",
    ),
    "mrad": ModuleSpec(
        name="mrad",
        kind="candidate_generator",
        source="MRAD baseline",
        baseline_path="../baselines/MRAD",
        weight_keys=["clip_vit_l_14"],
        command_hint="run upstream MRAD retrieval scoring and convert top regions to proposals",
    ),
    "exact_knn": ModuleSpec(
        name="exact_knn",
        kind="memory_backend",
        source="cera_iad.core.memory.NormalMemory",
        command_hint="load normal memory from .npy",
        notes="Default deterministic memory backend.",
    ),
    "faiss_knn": ModuleSpec(
        name="faiss_knn",
        kind="memory_backend",
        source="faiss-gpu/faiss-cpu optional dependency",
        command_hint="build FAISS index from normal memory features",
        notes="Large-scale ablation; falls back to exact_knn if unavailable.",
    ),
    "no_mask": ModuleSpec(
        name="no_mask",
        kind="mask_refiner",
        source="identity",
        notes="Disables geometry evidence.",
    ),
    "sam_box": ModuleSpec(
        name="sam_box",
        kind="mask_refiner",
        source="SAM box prompt",
        weight_keys=["sam_vit_h"],
        command_hint="refine proposal boxes with SAM masks",
    ),
    "sam2_box": ModuleSpec(
        name="sam2_box",
        kind="mask_refiner",
        source="SAM2.1 box prompt",
        weight_keys=["sam2_1_hiera_large"],
        command_hint="refine proposal boxes with SAM2 masks",
    ),
    "conformal_global": ModuleSpec(
        name="conformal_global",
        kind="calibrator",
        source="cera_iad.core.conformal.ConformalGate",
        command_hint="fit threshold from normal calibration scores",
    ),
    "fixed_threshold": ModuleSpec(
        name="fixed_threshold",
        kind="calibrator",
        source="fixed score threshold",
        notes="Ablation for conformal gate.",
    ),
    "linear_fusion": ModuleSpec(
        name="linear_fusion",
        kind="fusion_scorer",
        source="cera_iad.core.scoring.fuse_scores",
        notes="Default Semantic + Rarity + Disagreement + Uncertainty fusion.",
    ),
    "qwen25_vl": ModuleSpec(
        name="qwen25_vl",
        kind="reasoner",
        source="Hugging Face Qwen/Qwen2.5-VL-7B-Instruct",
        weight_keys=["qwen25_vl_7b"],
        command_hint="consume evidence JSON, crops, masks, and retrieved patches to produce reflection JSON",
    ),
    "internvl25": ModuleSpec(
        name="internvl25",
        kind="reasoner",
        source="Hugging Face OpenGVLab/InternVL2_5-8B",
        weight_keys=["internvl25_8b"],
        command_hint="optional MLLM reflection ablation",
    ),
    "no_reasoner": ModuleSpec(
        name="no_reasoner",
        kind="reasoner",
        source="identity",
        notes="Disables MLLM reflection.",
    ),
}


def get_module(name: str) -> ModuleSpec:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown module {name!r}; available: {choices}") from exc


def registry_snapshot() -> dict[str, dict[str, Any]]:
    return {name: spec.to_dict() for name, spec in sorted(REGISTRY.items())}


def build_experiment_plan(config: dict[str, Any], ablation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve configured module names into a serializable dry-run plan."""

    modules = dict(config.get("modules", {}))
    scoring = dict(config.get("scoring", {}))
    output_subdir = None
    if ablation:
        modules.update(ablation.get("modules", {}))
        scoring_override = ablation.get("scoring_override")
        if isinstance(scoring_override, dict):
            scoring.update(scoring_override)
        output_subdir = ablation.get("output_subdir")

    resolved: dict[str, Any] = {}
    required_weights: set[str] = set()
    for role, module_name in modules.items():
        spec = get_module(str(module_name))
        resolved[role] = spec.to_dict()
        required_weights.update(spec.weight_keys)

    return {
        "method": config.get("method", "CERA-IAD"),
        "ablation": None if ablation is None else ablation.get("name"),
        "modules": resolved,
        "required_weights": sorted(required_weights),
        "paths": config.get("paths", {}),
        "data": config.get("data", {}),
        "artifacts": config.get("artifacts", {}),
        "outputs": config.get("outputs", {}),
        "output_subdir": output_subdir,
        "scoring": scoring,
        "execution_note": "Use --dry-run to validate configuration before running full adapter commands.",
    }


def assert_baseline_paths_exist(repo_root: str | Path) -> list[str]:
    """Return missing baseline adapter paths without importing heavy baselines."""

    root = Path(repo_root)
    missing: list[str] = []
    for spec in REGISTRY.values():
        if spec.baseline_path and not (root / spec.baseline_path).exists():
            missing.append(spec.baseline_path)
    return sorted(set(missing))
