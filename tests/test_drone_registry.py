"""Tests for the drone registry and config wiring.

These tests mock Isaac Sim / Isaac Lab dependencies so they can run in a
plain Python environment (no simulator required).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Stub out heavy Isaac Sim / Isaac Lab imports before touching project code.
# ---------------------------------------------------------------------------

class _AttrModule(types.ModuleType):
    """Module that returns a MagicMock for any attribute that doesn't exist,
    and supports sub-package imports (e.g. ``import omni.ext``)."""

    def __init__(self, name, *args, **kwargs):
        super().__init__(name)
        self.__path__ = []  # makes it a package
        self.__file__ = f"<stub:{name}>"
        self.__all__ = []

    def __call__(self, *args, **kwargs):
        return mock.MagicMock()

    def __or__(self, other):
        """Support ``X | Y`` union type annotation syntax (PEP 604)."""
        return mock.MagicMock()

    def __ror__(self, other):
        return mock.MagicMock()

    def __init_subclass__(cls, **kwargs):
        """Allow subclassing stubs (e.g. ``class Foo(omni.ext.IExt)``)."""
        pass

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        # Auto-create child module stub for subpackage access
        child_name = f"{self.__name__}.{item}"
        if child_name in sys.modules:
            return sys.modules[child_name]
        child = _AttrModule(child_name)
        sys.modules[child_name] = child
        return child


_STUB_ROOTS = [
    "isaaclab",
    "isaaclab_tasks",
    "isaacsim",
    "carb",
    "omni",
    "tensordict",
    "gymnasium",
    "skrl",
    "wandb",
]

# Specific sub-modules we know are imported
_STUB_LEAVES = [
    "isaaclab.sim",
    "isaaclab.sim.schemas",
    "isaaclab.sim.converters",
    "isaaclab.sim.spawners",
    "isaaclab.sim.spawners.sensors",
    "isaaclab.sim.spawners.sensors.sensors_cfg",
    "isaaclab.actuators",
    "isaaclab.assets",
    "isaaclab.envs",
    "isaaclab.markers",
    "isaaclab.scene",
    "isaaclab.sensors",
    "isaaclab.sensors.camera",
    "isaaclab.sensors.camera.utils",
    "isaaclab.terrains",
    "isaaclab.utils",
    "isaaclab.utils.assets",
    "isaaclab.utils.math",
    "isaacsim.core",
    "isaacsim.core.utils",
    "isaacsim.core.utils.prims",
    "omni.ext",
]


@pytest.fixture(autouse=True)
def _patch_isaac_imports():
    """Temporarily install stubs for all Isaac / GPU dependencies."""
    saved = {}
    all_names = _STUB_ROOTS + _STUB_LEAVES

    for name in all_names:
        saved[name] = sys.modules.get(name, _SENTINEL)
        sys.modules[name] = _AttrModule(name)

    # Keep real yaml
    real_yaml = importlib.import_module("yaml")
    sys.modules["yaml"] = real_yaml

    # ISAAC_NUCLEUS_DIR needs to be a string
    sys.modules["isaaclab.utils.assets"].ISAAC_NUCLEUS_DIR = "/mock/nucleus"

    # UrdfConverterCfg needs nested attrs
    urdf_mock = mock.MagicMock()
    urdf_mock.JointDriveCfg = mock.MagicMock()
    urdf_mock.JointDriveCfg.PDGainsCfg = mock.MagicMock()
    sys.modules["isaaclab.sim.converters"].UrdfConverterCfg = urdf_mock

    yield

    # Restore originals
    for name in all_names:
        prev = saved[name]
        if prev is _SENTINEL:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev

    # Purge cached project modules so next test gets a fresh import
    to_remove = [k for k in sys.modules if k.startswith("source.isaac_pursuit_evasion")]
    for k in to_remove:
        del sys.modules[k]


_SENTINEL = object()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_registry():
    """Import the registry module (after stubs are in place)."""
    from source.isaac_pursuit_evasion.assets import drone_registry
    return drone_registry


def _load_module_directly(module_name: str, file_path: Path):
    """Load a single .py file as a module, bypassing __init__.py chains."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_env_module():
    """Import pursuit_evasion_env.py directly, bypassing isaac_pursuit_evasion __init__."""
    path = (
        _PROJECT_ROOT
        / "source/isaac_pursuit_evasion/isaac_pursuit_evasion/tasks/direct/pursuit_evasion/pursuit_evasion_env.py"
    )
    return _load_module_directly(
        "source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pursuit_evasion.pursuit_evasion_env",
        path,
    )


def _import_cfg_module():
    """Import pursuit_evasion_cfg.py directly."""
    # Ensure env module is loaded first (cfg imports from it)
    _import_env_module()
    path = (
        _PROJECT_ROOT
        / "source/isaac_pursuit_evasion/isaac_pursuit_evasion/tasks/direct/pursuit_evasion/pursuit_evasion_cfg.py"
    )
    return _load_module_directly(
        "source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pursuit_evasion.pursuit_evasion_cfg",
        path,
    )


# ---------------------------------------------------------------------------
# Tests: drone_registry.py
# ---------------------------------------------------------------------------

class TestDroneRegistry:
    def test_builtin_drones_registered(self):
        reg = _import_registry()
        names = reg.available_drones()
        assert "crazyflie_brushless" in names
        assert "crazyflie" in names
        assert "vaporx5" in names

    def test_aliases_resolve(self):
        reg = _import_registry()
        assert reg.get_drone_config("cf_brushless").name == "crazyflie_brushless"
        assert reg.get_drone_config("cf2x").name == "crazyflie"
        assert reg.get_drone_config("vapor_x5").name == "vaporx5"
        assert reg.get_drone_config("vapor").name == "vaporx5"

    def test_case_insensitive_lookup(self):
        reg = _import_registry()
        cfg1 = reg.get_drone_config("Crazyflie_Brushless")
        cfg2 = reg.get_drone_config("crazyflie_brushless")
        assert cfg1.name == cfg2.name

    def test_unknown_drone_raises(self):
        reg = _import_registry()
        with pytest.raises(ValueError, match="Unknown drone"):
            reg.get_drone_config("nonexistent_drone")

    def test_drone_config_has_required_fields(self):
        reg = _import_registry()
        for name in ("crazyflie_brushless", "crazyflie", "vaporx5"):
            dc = reg.get_drone_config(name)
            assert dc.pursuer_cfg is not None
            assert dc.evader_cfg is not None
            assert isinstance(dc.body_name, str) and dc.body_name
            assert isinstance(dc.prop_joint_patterns, list) and len(dc.prop_joint_patterns) > 0
            assert dc.fpv_camera_cfg_fn is not None and callable(dc.fpv_camera_cfg_fn)
            assert dc.fpv_camera_center_line_fn is not None and callable(dc.fpv_camera_center_line_fn)
            assert dc.transform_camera_line_fn is not None and callable(dc.transform_camera_line_fn)
            assert dc.dynamics_name is not None

    def test_register_custom_drone(self):
        reg = _import_registry()
        custom = reg.DroneConfig(
            name="testdrone",
            pursuer_cfg=mock.MagicMock(),
            evader_cfg=mock.MagicMock(),
            body_name="base_link",
            aliases=("td",),
        )
        reg.register_drone(custom)
        assert reg.get_drone_config("testdrone") is custom
        assert reg.get_drone_config("td") is custom

    def test_prop_joint_patterns_are_list_of_lists(self):
        reg = _import_registry()
        for name in ("crazyflie_brushless", "crazyflie", "vaporx5"):
            dc = reg.get_drone_config(name)
            for pattern_group in dc.prop_joint_patterns:
                assert isinstance(pattern_group, list)
                for p in pattern_group:
                    assert isinstance(p, str)


# ---------------------------------------------------------------------------
# Tests: pursuit_evasion_cfg.py wiring
# ---------------------------------------------------------------------------

class TestCfgDroneNameWiring:
    """Verify that config builder functions propagate drone_name correctly."""

    def _patched_cfg_module(self):
        cfg_mod = _import_cfg_module()
        # Patch load_controller_config at the module level where it was imported
        m = mock.MagicMock(return_value={"mocked": True})
        cfg_mod.load_controller_config = m
        return cfg_mod, m

    def test_pretrain_frpn_sets_drone_name(self):
        cfg_mod, _ = self._patched_cfg_module()
        cfg = cfg_mod.pretrain_frpn_vs_rl_cfg(num_envs=4, drone_name="crazyflie")
        assert cfg.drone_name == "crazyflie"
        # Controller config should NOT be pre-loaded (deferred to env init)
        frpn_spec = cfg.pursuer_controllers[0]
        assert frpn_spec.config is None
        assert frpn_spec.config_overrides is not None
        assert "curriculum" in frpn_spec.config_overrides

    def test_bench_frpn_vs_apf_sets_drone_name(self):
        cfg_mod, _ = self._patched_cfg_module()
        cfg = cfg_mod.bench_frpn_vs_apf_cfg(num_envs=4, drone_name="vaporx5")
        assert cfg.drone_name == "vaporx5"
        # No pre-loaded controller config
        assert cfg.pursuer_controllers[0].config is None
        assert cfg.evader_controllers[0].config is None

    def test_bench_slowfrpn_sets_drone_name(self):
        cfg_mod, _ = self._patched_cfg_module()
        cfg = cfg_mod.bench_slowfrpn_vs_apf_cfg(num_envs=4, drone_name="crazyflie")
        assert cfg.drone_name == "crazyflie"
        assert cfg.pursuer_controllers[0].config is None
        assert cfg.evader_controllers[0].config is None

    def test_bench_frpn_vs_hover_sets_drone_name(self):
        cfg_mod, _ = self._patched_cfg_module()
        cfg = cfg_mod.bench_frpn_vs_hover_cfg(num_envs=4, drone_name="vaporx5")
        assert cfg.drone_name == "vaporx5"

    def test_bench_frpn_vs_trajectories_sets_drone_name(self):
        cfg_mod, _ = self._patched_cfg_module()
        cfg = cfg_mod.bench_frpn_vs_trajectories_cfg(num_envs=4, drone_name="crazyflie")
        assert cfg.drone_name == "crazyflie"

    def test_bench_rl_vs_apf_sets_drone_name(self):
        cfg_mod, _ = self._patched_cfg_module()
        cfg = cfg_mod.bench_rl_vs_apf_cfg(num_envs=4, drone_name="vaporx5")
        assert cfg.drone_name == "vaporx5"

    def test_default_drone_name_is_crazyflie_brushless(self):
        """Verify the default drone_name value from source code."""
        src = (
            _PROJECT_ROOT
            / "source/isaac_pursuit_evasion/isaac_pursuit_evasion/tasks/direct/pursuit_evasion/pursuit_evasion_cfg.py"
        ).read_text()
        assert 'drone_name: str = "crazyflie_brushless"' in src

    def test_ablation_config_sets_drone_name_crazyflie(self):
        cfg_mod, _ = self._patched_cfg_module()
        cfg = cfg_mod.ablation_vision_vs_trajectories_cfg(num_envs=4)
        assert cfg.drone_name == "crazyflie"

    def test_ablation_action_mode_override(self):
        """Verify action mode can be overridden on the ablation config."""
        cfg_mod, _ = self._patched_cfg_module()
        cfg = cfg_mod.ablation_vision_vs_trajectories_cfg(num_envs=4)
        # Default is body rates
        assert cfg.agent_action_mode == "body_rates"
        assert cfg.pursuer_controllers[0].kind == "rl_bodyrates"
        # Simulate --action-mode=rl_velocity override (same logic as train.py)
        cfg.agent_action_mode = "velocity"
        for spec in cfg.pursuer_controllers + cfg.evader_controllers:
            if hasattr(spec, "kind") and spec.kind in ("rl_bodyrates", "rl_velocity"):
                spec.kind = "rl_velocity"
        assert cfg.agent_action_mode == "velocity"
        assert cfg.pursuer_controllers[0].kind == "rl_velocity"

    def test_ppo_rnn_asym_in_runner_guard(self):
        """Verify PPO_RNN_ASYM is handled by _generate_agent in runner.py."""
        runner_path = (
            _PROJECT_ROOT
            / "source/isaac_pursuit_evasion/isaac_pursuit_evasion/skrl_ext/runner.py"
        )
        src = runner_path.read_text()
        # The guard list must include ppo_rnn_asym
        assert '"ppo_rnn_asym"' in src

    def test_symmetric_critic_skips_state_space_swap(self):
        """Verify _generate_models only swaps state_space for asymmetric agent classes.

        The symmetric PPO variant (class: PPO) must use the dict observation_space
        for both actor and critic, not the flat state_space.
        """
        runner_path = (
            _PROJECT_ROOT
            / "source/isaac_pursuit_evasion/isaac_pursuit_evasion/skrl_ext/runner.py"
        )
        src = runner_path.read_text()
        # The _generate_models method must check agent_class before swapping
        assert "agent_class in _asym_agents" in src
        # The set must include all three asymmetric agent classes
        assert '"ppo_asym"' in src
        assert '"ppo_rnn_vsh"' in src
        assert '"ppo_rnn_asym"' in src


# ---------------------------------------------------------------------------
# Tests: env source code inspection (no instantiation needed)
# ---------------------------------------------------------------------------

class TestEnvDroneSpecWiring:
    """Verify that the env source uses drone_spec fields correctly."""

    def _read_env_source(self):
        path = (
            _PROJECT_ROOT
            / "source/isaac_pursuit_evasion/isaac_pursuit_evasion/tasks/direct/pursuit_evasion/pursuit_evasion_env.py"
        )
        return path.read_text()

    def test_env_init_uses_get_drone_config(self):
        src = self._read_env_source()
        assert "get_drone_config" in src
        assert "drone_spec.pursuer_cfg" in src
        assert "drone_spec.evader_cfg" in src
        assert "drone_spec.body_name" in src
        assert "drone_spec.fpv_camera_cfg_fn" in src
        assert "drone_spec.fpv_camera_center_line_fn" in src

    def test_find_prop_joints_uses_drone_spec_patterns(self):
        src = self._read_env_source()
        assert "self._drone_spec.prop_joint_patterns" in src
        # Should NOT contain old hardcoded patterns list
        assert 'patterns_to_try = [\n            ["revolute_prop_.*"]' not in src

    def test_spawn_fpv_cameras_uses_drone_spec(self):
        src = self._read_env_source()
        assert "self._drone_spec.fpv_camera_cfg_fn" in src

    def test_no_hardcoded_drone_imports_in_env(self):
        src = self._read_env_source()
        assert "CrazyflieBrushlessPursuer" not in src
        assert "CrazyflieBrushlessEvader" not in src
        # The import should be from drone_registry
        assert "from source.isaac_pursuit_evasion.assets.drone_registry import get_drone_config" in src

    def test_transform_camera_line_uses_drone_spec(self):
        src = self._read_env_source()
        assert "self._drone_spec.transform_camera_line_fn" in src

    def test_env_imports_module_directly(self):
        """Verify env module can be loaded directly without errors."""
        env_mod = _import_env_module()
        assert hasattr(env_mod, "PursuitEvasionEnv")
        assert hasattr(env_mod, "PursuitEvasionEnvCfg")
