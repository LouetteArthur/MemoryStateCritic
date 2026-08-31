# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""pytest configuration for running without Isaac Sim / Isaac Lab.

Sets up two things:
1. sys.path: adds project root so ``from source.isaac_pursuit_evasion.* import ...`` works.
2. _IsaacStubFinder: a meta-path finder installed at session level that
   intercepts ANY import attempt for Isaac Lab / Isaac Sim packages and
   returns a stub module.  This is required because Python's import
   machinery looks up submodules (e.g. ``isaaclab_tasks.utils``) via
   sys.meta_path, not via the parent module's ``__getattr__``, so the
   per-test ``_patch_isaac_imports`` fixture alone cannot handle them.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

# ---------------------------------------------------------------------------
# 1. Project root on sys.path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 2. Meta-path finder for Isaac Lab / Isaac Sim stubs
# ---------------------------------------------------------------------------

_STUB_PREFIXES = (
    "isaaclab",
    "isaaclab_tasks",
    "isaaclab_assets",
    "isaaclab_rl",
    "isaacsim",
    "carb",
    "omni",
    "wandb",
    "tensordict",
)


class _IsaacStubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__file__ = f"<stub:{spec.name}>"
        mod.__all__ = []
        return mod

    def exec_module(self, module):
        def __getattr__(item):
            if item.startswith("__") and item.endswith("__"):
                raise AttributeError(item)
            child_name = f"{module.__name__}.{item}"
            if child_name not in sys.modules:
                child = mock.MagicMock()
                sys.modules[child_name] = child
            return sys.modules[child_name]

        module.__getattr__ = __getattr__
        module.__call__ = lambda *a, **kw: mock.MagicMock()


class _IsaacStubFinder(importlib.abc.MetaPathFinder):
    _loader = _IsaacStubLoader()

    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in _STUB_PREFIXES):
            return importlib.util.spec_from_loader(fullname, self._loader)
        return None


_stub_finder = _IsaacStubFinder()
if not any(isinstance(f, _IsaacStubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _stub_finder)
