from __future__ import annotations

from pathlib import Path


class CloudAdapterCommand:
    """Command template for upstream baseline integration.

    This object deliberately does not execute commands. The experiment runner
    uses it to document and assemble steps after datasets and weights exist.
    """

    def __init__(self, name: str, baseline_dir: str | None, command: list[str]) -> None:
        self.name = name
        self.baseline_dir = baseline_dir
        self.command = command

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "baseline_dir": self.baseline_dir,
            "command": self.command,
        }


def proposal_converter_command(
    module_name: str,
    baseline_dir: str | None,
    input_dir: str,
    output_json: str,
) -> CloudAdapterCommand:
    """Return the intended conversion command for proposal adapters."""

    return CloudAdapterCommand(
        name=f"{module_name}_to_region_proposals",
        baseline_dir=baseline_dir,
        command=[
            "python",
            "scripts/run_cera.py",
            "--adapter-only",
            module_name,
            "--input-dir",
            input_dir,
            "--output",
            output_json,
        ],
    )


def ensure_path_string(path: str | Path) -> str:
    return str(Path(path).as_posix())
