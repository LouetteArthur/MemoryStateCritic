"""Lightweight actor-only policy loader for deployment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from source.isaac_pursuit_evasion.deployment.policy_loader_utils import (
    download_wandb_artifact,
    get_activation,
    load_config_data,
    resolve_cfg_path,
    strip_prefix,
)
from source.isaac_pursuit_evasion.deployment.skrl_scaler import load_skrl_scalers


_CFG_DIR = Path(__file__).parent / "cfg"


@dataclass
class ActorPolicyConfig:
    obs_dim: int
    action_dim: int
    hidden_layers: Sequence[int]
    activation: str = "elu"
    log_std_init: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActorPolicyConfig":
        return cls(
            obs_dim=int(data["obs_dim"]),
            action_dim=int(data["action_dim"]),
            hidden_layers=[int(x) for x in data.get("hidden_layers", ())],
            activation=str(data.get("activation", "elu")).lower(),
            log_std_init=float(data.get("log_std_init", 0.0)),
        )


class SimpleGaussianActor(nn.Module):
    """Minimal MLP actor matching skrl Gaussian model naming (net_container + log_std_parameter)."""

    def __init__(self, cfg: ActorPolicyConfig) -> None:
        super().__init__()
        act_cls = get_activation(cfg.activation)
        layers: list[nn.Module] = []
        in_dim = cfg.obs_dim
        for hidden in cfg.hidden_layers:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(act_cls())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.action_dim))
        self.net_container = nn.Sequential(*layers)
        self.log_std_parameter = nn.Parameter(
            torch.full((cfg.action_dim,), float(cfg.log_std_init), dtype=torch.float32)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net_container(obs)

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mean = self.forward(obs)
        if deterministic:
            return mean
        std = torch.exp(self.log_std_parameter).expand_as(mean)
        return mean + std * torch.randn_like(mean)


def _extract_policy_state_dict(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, nn.Module):
        return payload.state_dict()
    if not isinstance(payload, Mapping):
        return None

    if "policy" in payload and isinstance(payload["policy"], Mapping):
        return payload["policy"]
    for container_key in ("models", "model", "model_state_dict", "state_dict"):
        container = payload.get(container_key)
        if isinstance(container, Mapping):
            if "policy" in container and isinstance(container["policy"], Mapping):
                return container["policy"]
            for prefix in ("policy", "models.policy", "model.policy"):
                filtered = strip_prefix(container, prefix)
                if filtered:
                    return filtered

    for prefix in ("policy", "models.policy", "model.policy"):
        filtered = strip_prefix(payload, prefix)
        if filtered:
            return filtered

    if any(key.startswith("net_container.") for key in payload.keys()):
        return payload
    return None


def load_actor_policy_config(path: str | Path | None = None) -> ActorPolicyConfig:
    path = resolve_cfg_path(path, "actor_tracker_cfg.yml", _CFG_DIR)
    data = load_config_data(path)
    return ActorPolicyConfig.from_dict(data)


def load_actor_from_checkpoint(
    checkpoint: str | Path,
    cfg: ActorPolicyConfig,
    *,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> SimpleGaussianActor:
    checkpoint = str(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = _extract_policy_state_dict(payload)
    if state_dict is None:
        raise ValueError(f"Unable to locate policy weights in checkpoint: {checkpoint}")

    actor = SimpleGaussianActor(cfg)
    missing, unexpected = actor.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(f"[WARN] Actor checkpoint load mismatch (missing={missing}, unexpected={unexpected}).")
    obs_scaler, value_scaler = load_skrl_scalers(payload)
    actor.obs_scaler = obs_scaler
    actor.value_scaler = value_scaler
    actor.to(device)
    actor.eval()
    return actor


def load_actor_from_wandb(
    artifact: str,
    *,
    artifact_file: str | None = None,
    local_dir: str | Path | None = None,
    cfg: ActorPolicyConfig | None = None,
    device: str | torch.device = "cpu",
) -> SimpleGaussianActor:
    cfg = cfg or load_actor_policy_config()
    checkpoint = download_wandb_artifact(artifact, artifact_file, local_dir)
    return load_actor_from_checkpoint(checkpoint, cfg, device=device)


def load_tracker_cf_rate_actor(
    *,
    device: str | torch.device = "cpu",
    artifact: str = "kthxulg/ppo_baseline/pretrain_tracker_cf_rate_DR_NoNorm:latest",
    artifact_file: str | None = None,
    cfg_path: str | Path | None = None,
) -> SimpleGaussianActor:
    cfg = load_actor_policy_config(cfg_path)
    return load_actor_from_wandb(
        artifact,
        artifact_file=artifact_file,
        cfg=cfg,
        device=device,
    )


class ActorPolicyCallable:
    """Adapter to use a SimpleGaussianActor with RL wrappers (TensorDict in, Tensor out)."""

    def __init__(self, actor: SimpleGaussianActor, device: str | torch.device = "cpu") -> None:
        self.actor = actor
        self.device = torch.device(device)
        self.actor.to(self.device)
        self.actor.eval()

    def __call__(self, td) -> torch.Tensor:
        obs = td.get("observation")
        if obs is None:
            raise KeyError("Expected 'observation' key in TensorDict for actor policy execution.")
        obs = obs.to(self.device)
        with torch.no_grad():
            return self.actor.act(obs, deterministic=True)


# ---------------------------------------------------------------------------
# Recurrent (CNN+GRU) actor for vision-based opponent loading
# ---------------------------------------------------------------------------

@dataclass
class RecurrentActorConfig:
    """Configuration for loading a CNN+GRU recurrent actor (GaussianCNNGRUModel)."""

    image_channels: int
    image_height: int
    image_width: int
    past_actions_size: int
    action_dim: int
    cnn_feature_size: int = 128
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecurrentActorConfig:
        return cls(
            image_channels=int(data["image_channels"]),
            image_height=int(data["image_height"]),
            image_width=int(data["image_width"]),
            past_actions_size=int(data["past_actions_size"]),
            action_dim=int(data["action_dim"]),
            cnn_feature_size=int(data.get("cnn_feature_size", 128)),
            rnn_hidden_size=int(data.get("rnn_hidden_size", 256)),
            rnn_num_layers=int(data.get("rnn_num_layers", 1)),
        )


class SimpleRecurrentActor(nn.Module):
    """Standalone CNN+GRU actor matching GaussianCNNGRUModel architecture.

    Weight keys are compatible: cnn.*, cnn_linear.*, input_ln.*, rnn.*,
    output_layer.*, log_std_parameter.
    """

    def __init__(self, cfg: RecurrentActorConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.cnn = nn.Sequential(
            nn.Conv2d(cfg.image_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size
        dummy = torch.zeros(1, cfg.image_channels, cfg.image_height, cfg.image_width)
        cnn_out_size = self.cnn(dummy).shape[-1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_out_size, cfg.cnn_feature_size),
            nn.ELU(),
        )

        gru_input_size = cfg.cnn_feature_size + cfg.past_actions_size
        self.input_ln = nn.LayerNorm(gru_input_size)

        self.rnn = nn.GRU(
            input_size=gru_input_size,
            hidden_size=cfg.rnn_hidden_size,
            num_layers=cfg.rnn_num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(cfg.rnn_hidden_size, cfg.action_dim)
        self.log_std_parameter = nn.Parameter(torch.zeros(cfg.action_dim))

    def forward(
        self, image: torch.Tensor, past_actions: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward: (batch, C, H, W) + (batch, A) + h -> action, h_new."""
        cnn_out = self.cnn(image)
        cnn_features = self.cnn_linear(cnn_out)

        if self.cfg.past_actions_size > 0:
            features = torch.cat([cnn_features, past_actions.reshape(cnn_features.shape[0], -1)], dim=-1)
        else:
            features = cnn_features
        features = self.input_ln(features)

        x = features.unsqueeze(1)  # (batch, 1, input_size)
        out, h_new = self.rnn(x, h.contiguous())
        action = self.output_layer(out.squeeze(1))
        return action, h_new


def load_recurrent_actor_from_checkpoint(
    checkpoint: str | Path,
    cfg: RecurrentActorConfig,
    *,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> SimpleRecurrentActor:
    """Load a CNN+GRU actor from a training checkpoint."""
    payload = torch.load(str(checkpoint), map_location="cpu")
    state_dict = _extract_policy_state_dict(payload)
    if state_dict is None:
        raise ValueError(f"Unable to locate policy weights in checkpoint: {checkpoint}")

    actor = SimpleRecurrentActor(cfg)
    missing, unexpected = actor.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(f"[WARN] Recurrent actor checkpoint mismatch (missing={missing}, unexpected={unexpected}).")
    actor.to(device)
    actor.eval()
    return actor


class RecurrentActorPolicyCallable:
    """Adapter for CNN+GRU actors as opponent policies.

    Maintains GRU hidden state across steps and exposes it via ``get_z()``.
    Expects TensorDict with ``image`` (C, H, W) and ``past_actions`` keys.
    """

    def __init__(
        self, actor: SimpleRecurrentActor, num_envs: int, device: str | torch.device = "cpu"
    ) -> None:
        self.actor = actor
        self.device = torch.device(device)
        self.actor.to(self.device)
        self.actor.eval()
        self._num_envs = num_envs
        self._h = torch.zeros(
            actor.cfg.rnn_num_layers,
            num_envs,
            actor.cfg.rnn_hidden_size,
            device=self.device,
        )

    @property
    def is_recurrent(self) -> bool:
        return True

    @property
    def rnn_hidden_size(self) -> int:
        return self.actor.cfg.rnn_hidden_size

    def __call__(self, td) -> torch.Tensor:
        """Run forward pass, update hidden state, return action."""
        image = td.get("image")
        past_actions = td.get("past_actions")
        if image is None or past_actions is None:
            raise KeyError("RecurrentActorPolicyCallable expects 'image' and 'past_actions' in TensorDict.")
        image = image.to(self.device)
        past_actions = past_actions.to(self.device).reshape(image.shape[0], -1)

        with torch.no_grad():
            action, h_new = self.actor(image, past_actions, self._h[:, : image.shape[0]])
            self._h[:, : image.shape[0]] = h_new

        # Defensive sanitization: the actor's output is the unbounded
        # Gaussian mean. If a corrupt checkpoint is ever loaded (e.g. the
        # original pretrain runs that diverged to all-NaN weights) the
        # clamp alone would leave NaN as NaN — nan_to_num neutralises it.
        return torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

    def get_z(self) -> torch.Tensor:
        """Return current GRU hidden state (last layer, all envs, detached)."""
        return self._h[-1].detach()

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset hidden state for specific environments (or all if None)."""
        if env_ids is None:
            self._h.zero_()
        else:
            self._h[:, env_ids] = 0.0
