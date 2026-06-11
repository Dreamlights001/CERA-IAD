"""Config-driven module registry for CERA-IAD experiments."""

from cera_iad.modules.registry import ModuleSpec, build_experiment_plan, registry_snapshot

__all__ = ["ModuleSpec", "build_experiment_plan", "registry_snapshot"]
