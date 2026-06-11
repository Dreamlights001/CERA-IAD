from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cera_iad_config.yaml"
DEFAULT_MANIFEST = PROJECT_ROOT / "weights" / "weights_manifest.yaml"
AUTOMATIC_GROUPS = {"basic", "optional", "mllm"}


@dataclass(frozen=True)
class WeightSpec:
    key: str
    display_name: str
    group: str
    hf_repo: str | None
    target_dir: str
    required_for: list[str]
    required: bool
    size_hint: str
    license_note: str
    note: str

    @property
    def is_manual(self) -> bool:
        return not self.hf_repo or self.group == "manual"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify CERA-IAD pretrained checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--basic", action="store_const", const="basic", dest="mode", help="Download CLIP and SAM.")
    mode.add_argument("--optional", action="store_const", const="optional", dest="mode", help="Download optional ablation weights.")
    mode.add_argument("--mllm", action="store_const", const="mllm", dest="mode", help="Download MLLM reasoner weights.")
    mode.add_argument("--all", action="store_const", const="all", dest="mode", help="Download all automatic weights.")
    parser.set_defaults(mode="basic")
    parser.add_argument("--list", action="store_true", help="List weights and local availability without downloading.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected download plan without network access.")
    parser.add_argument("--force", action="store_true", help="Remove selected local weight directories before downloading.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Main CERA-IAD config.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Weight manifest.")
    return parser.parse_args(argv)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def resolve_path(path: Path, base: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def load_weights_root(config_path: Path) -> Path:
    config = read_yaml(resolve_path(config_path))
    weights_root = Path(config.get("paths", {}).get("weights_root", "weights/cache"))
    return resolve_path(weights_root)


def load_manifest(manifest_path: Path) -> list[WeightSpec]:
    manifest = read_yaml(resolve_path(manifest_path))
    models = manifest.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("weights manifest must contain a 'models' mapping")

    specs: list[WeightSpec] = []
    for key, raw in models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"invalid model entry for {key}")
        specs.append(
            WeightSpec(
                key=str(key),
                display_name=str(raw.get("display_name") or key),
                group=str(raw.get("group") or "manual"),
                hf_repo=raw.get("hf_repo"),
                target_dir=str(raw.get("target_dir") or key),
                required_for=list(raw.get("required_for") or []),
                required=bool(raw.get("required", False)),
                size_hint=str(raw.get("size_hint") or "not specified"),
                license_note=str(raw.get("license_note") or "See the upstream model card."),
                note=str(raw.get("note") or ""),
            )
        )
    return specs


def selected_specs(specs: list[WeightSpec], mode: str) -> list[WeightSpec]:
    if mode == "all":
        return [spec for spec in specs if spec.group in AUTOMATIC_GROUPS]
    return [spec for spec in specs if spec.group == mode]


def metadata_path(weights_root: Path, spec: WeightSpec) -> Path:
    return weights_root / spec.target_dir / ".cera_weight.json"


def local_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.name != ".cera_weight.json")


def status_for(weights_root: Path, spec: WeightSpec) -> str:
    local_dir = weights_root / spec.target_dir
    meta = metadata_path(weights_root, spec)
    files_count = local_file_count(local_dir)
    if meta.exists() and files_count > 0:
        return "available"
    if local_dir.exists() and files_count > 0:
        return "partial"
    if spec.is_manual:
        return "manual"
    return "missing"


def print_table(specs: list[WeightSpec], weights_root: Path) -> None:
    print(f"[CERA-IAD] weights root: {weights_root}")
    print("[CERA-IAD] pretrained weight registry")
    for spec in specs:
        target = weights_root / spec.target_dir
        repo = spec.hf_repo or "manual"
        required_for = ", ".join(spec.required_for) if spec.required_for else "optional/manual"
        print(
            f"- {spec.key:<24} [{status_for(weights_root, spec):>9}] "
            f"{spec.display_name} | group={spec.group} | repo={repo}"
        )
        print(f"  target: {target}")
        print(f"  used by: {required_for}")
        print(f"  size: {spec.size_hint}")


def ensure_huggingface_hub() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install the project dependencies first:\n"
            "  python -m pip install -r requirements.txt\n"
            "or install it directly:\n"
            "  python -m pip install -U huggingface_hub"
        ) from exc
    return snapshot_download


def write_metadata(weights_root: Path, spec: WeightSpec) -> None:
    local_dir = weights_root / spec.target_dir
    files_count = local_file_count(local_dir)
    payload = {
        "key": spec.key,
        "display_name": spec.display_name,
        "group": spec.group,
        "hf_repo": spec.hf_repo,
        "target_dir": spec.target_dir,
        "download_time": datetime.now(timezone.utc).isoformat(),
        "required_for": spec.required_for,
        "local_files_count": files_count,
        "license_note": spec.license_note,
    }
    metadata_path(weights_root, spec).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def download_spec(index: int, total: int, spec: WeightSpec, weights_root: Path, force: bool) -> bool:
    if spec.is_manual:
        print(f"[{index}/{total}] {spec.display_name} | manual checkpoint")
        print(f"  skip: {spec.note}")
        return True

    local_dir = weights_root / spec.target_dir
    meta = metadata_path(weights_root, spec)
    if meta.exists() and local_file_count(local_dir) > 0 and not force:
        print(f"[{index}/{total}] {spec.display_name} | already available")
        print(f"  target: {local_dir}")
        return True

    if force and local_dir.exists():
        print(f"[{index}/{total}] {spec.display_name} | refreshing local files")
        shutil.rmtree(local_dir)
    else:
        print(f"[{index}/{total}] {spec.display_name} | {spec.hf_repo} -> {local_dir}")

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download = ensure_huggingface_hub()
    try:
        snapshot_download(
            repo_id=str(spec.hf_repo),
            local_dir=str(local_dir),
        )
    except Exception as exc:  # noqa: BLE001 - provide a user-facing diagnostic.
        print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  hint: check network access, Hugging Face login, disk space, and repo permissions.", file=sys.stderr)
        return False

    files_count = local_file_count(local_dir)
    if files_count == 0:
        print("  failed: no files were found after download", file=sys.stderr)
        return False
    write_metadata(weights_root, spec)
    print(f"  done: {files_count} files")
    print(f"  metadata: {meta}")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        weights_root = load_weights_root(args.config)
        specs = load_manifest(args.manifest)
        picked = selected_specs(specs, args.mode)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if args.list:
        print_table(specs, weights_root)
        return 0

    if not picked:
        print(f"[ERROR] no weights selected for mode: {args.mode}", file=sys.stderr)
        return 2

    weights_root.mkdir(parents=True, exist_ok=True)
    print(f"[CERA-IAD] weights root: {weights_root}")
    print(f"[CERA-IAD] mode: {args.mode}")
    print(f"[CERA-IAD] selected weights: {len(picked)}")

    if args.dry_run:
        for index, spec in enumerate(picked, start=1):
            repo = spec.hf_repo or "manual"
            print(f"[{index}/{len(picked)}] {spec.display_name} | {repo} -> {weights_root / spec.target_dir}")
            print(f"  status: {status_for(weights_root, spec)}")
        print("[CERA-IAD] dry run complete; no files were downloaded.")
        return 0

    failures = 0
    for index, spec in enumerate(picked, start=1):
        if not download_spec(index, len(picked), spec, weights_root, args.force):
            failures += 1

    if failures:
        print(f"[CERA-IAD] completed with {failures} failed download(s).", file=sys.stderr)
        return 1
    print("[CERA-IAD] all selected weights are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
