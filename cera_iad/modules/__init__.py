"""Config-driven module registry for cloud CERA-IAD experiments."""

from cera_iad.modules.registry import ModuleSpec, build_cloud_plan, registry_snapshot

__all__ = ["ModuleSpec", "build_cloud_plan", "registry_snapshot"]
