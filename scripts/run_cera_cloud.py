from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from cera_iad.modules.registry import assert_baseline_paths_exist, build_cloud_plan, registry_snapshot


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        value = _expand_bash_defaults(value)
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


_BASH_DEFAULT_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]+)\}")


def _expand_bash_defaults(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        env_name, default = match.group(1), match.group(2)
        return os.environ.get(env_name) or default

    return _BASH_DEFAULT_PATTERN.sub(replace, value)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return _expand_env(data)


def load_ablation(path: str | Path, name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    matrix = load_yaml(path)
    ablations = matrix.get("ablations", {})
    if name not in ablations:
        choices = ", ".join(sorted(ablations))
        raise KeyError(f"unknown ablation {name!r}; available: {choices}")
    return ablations[name]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud orchestrator for CERA-IAD experiments.")
    parser.add_argument("--config", default="configs/cera_iad_cloud.yaml")
    parser.add_argument("--ablation-matrix", default="configs/ablation_matrix.yaml")
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--output-plan", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Resolve modules and write plan only.")
    parser.add_argument("--print-registry", action="store_true")
    parser.add_argument("--adapter-only", default=None, help="Reserved cloud adapter mode for upstream baselines.")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.print_registry:
        print(json.dumps(registry_snapshot(), indent=2, ensure_ascii=False))
        return

    config = load_yaml(args.config)
    ablation = load_ablation(args.ablation_matrix, args.ablation)
    plan = build_cloud_plan(config, ablation)

    repo_root = Path(os.environ.get("IAD_ROOT", Path.cwd()))
    missing = assert_baseline_paths_exist(repo_root)
    if missing:
        plan["missing_baseline_paths"] = missing

    if args.adapter_only:
        plan["adapter_only"] = {
            "module": args.adapter_only,
            "input_dir": args.input_dir,
            "output": args.output,
            "status": "template_only",
        }

    if args.output_plan:
        write_json(args.output_plan, plan)

    print(json.dumps(plan, indent=2, ensure_ascii=False))

    if args.dry_run:
        return

    raise SystemExit(
        "Cloud execution hooks are scaffolded but disabled in this local implementation. "
        "Use --dry-run now; fill adapter commands on Ubuntu after datasets and weights are ready."
    )


if __name__ == "__main__":
    main()
