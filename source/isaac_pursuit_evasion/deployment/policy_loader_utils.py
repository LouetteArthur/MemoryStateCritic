"""Shared helpers for lightweight policy loaders."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch.nn as nn


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def get_activation(name: str | None) -> type[nn.Module]:
    if not name:
        return nn.ELU
    return ACTIVATIONS.get(str(name).lower(), nn.ELU)


def strip_prefix(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    token = f"{prefix}."
    return {key[len(token) :]: value for key, value in state_dict.items() if key.startswith(token)}


def resolve_cfg_path(path: str | Path | None, default_name: str, base_dir: Path) -> Path:
    if path is None:
        return base_dir / default_name
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute() and candidate.parent == Path("."):
        fallback = base_dir / candidate.name
        if fallback.exists():
            return fallback
    return candidate


def load_config_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy config not found at {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ImportError("PyYAML is required to parse policy YAML configs.") from exc
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return dict(data)
    import json

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data)


def download_wandb_artifact(
    artifact: str,
    artifact_file: str | None = None,
    local_dir: str | Path | None = None,
) -> str:
    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise ImportError("wandb is required to download artifacts.") from exc

    api = wandb.Api()
    artifact_obj = api.artifact(artifact)
    download_dir = Path(artifact_obj.download(root=str(local_dir)) if local_dir else artifact_obj.download())
    if artifact_file:
        candidate = download_dir / artifact_file
        if candidate.exists():
            return str(candidate)
    pt_files = sorted(download_dir.rglob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoints found in artifact {artifact}")
    return str(pt_files[-1])
