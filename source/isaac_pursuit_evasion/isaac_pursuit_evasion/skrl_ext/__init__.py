# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-local extensions to skrl.

This package contains all the custom algorithmic additions used by
isaac_pursuit_evasion that are not part of upstream skrl.  Keeping them
here (rather than patching the vendored skrl copy) makes it trivial to:

- diff exactly what is original contribution vs library code
- upgrade skrl without merge conflicts

Contents
--------
- ``models.gaussian_cnn_rnn``  : CNN+GRU actor with past-action concat
  (R2D2/IMPALA recipe), used for POMDPs with partial visual observability.
- ``models.memory_state_critic``  : the memory-state critic V(s, z^a) (ours),
  an MLP over the privileged state and the detached actor GRU hidden state.
- ``models.history_state_critic`` : the history-state critic V(s, z^c)
  (baseline), with its own CNN+GRU over the observation-action stream.
- ``agents.ppo_rnn_asym``      : recurrent PPO with an asymmetric critic; runs
  all four critics of the paper, selected by the YAML.
- ``runner``                   : ``CustomRunner`` that registers the
  models and agents above via the standard skrl YAML class names.

Import-time side effect
-----------------------
Importing this package monkey-patches
``skrl.utils.spaces.torch.unflatten_tensorized_space`` so it also accepts an
already-unflattened ``dict`` input, which is needed by the dict-shaped
observation-state critic path in ``PPO_RNN_ASYM``.  Upstream skrl only handles flat
tensor inputs, and we would rather patch at import time than fork the
vendored skrl copy.
"""

from gymnasium import spaces as _gym_spaces
from skrl.utils.spaces.torch import spaces as _skrl_spaces_module

_UNFLATTEN_PATCHED_FLAG = "_isaac_pe_dict_input_patch"


def _apply_unflatten_dict_patch() -> None:
    """Wrap ``unflatten_tensorized_space`` to short-circuit when ``x`` is a dict.

    The upstream implementation only handles flat tensor inputs when the space
    is ``spaces.Dict``.  Our dict-based unbiased critic passes an already
    unflattened dict through ``value.act({"states": dict})``, which then
    reaches the model's ``compute`` and calls ``unflatten_tensorized_space``
    again.  This wrapper detects that case and recurses per-key instead of
    crashing.
    """
    original = getattr(_skrl_spaces_module, "unflatten_tensorized_space")
    if getattr(original, _UNFLATTEN_PATCHED_FLAG, False):
        return  # already patched

    def unflatten_tensorized_space(space, x):  # type: ignore[override]
        if isinstance(space, _gym_spaces.Dict) and isinstance(x, dict):
            return {k: unflatten_tensorized_space(space[k], x[k]) for k in sorted(space.keys())}
        return original(space, x)

    setattr(unflatten_tensorized_space, _UNFLATTEN_PATCHED_FLAG, True)
    # patch the module attribute and the re-export at the package level
    _skrl_spaces_module.unflatten_tensorized_space = unflatten_tensorized_space
    import skrl.utils.spaces.torch as _skrl_spaces_pkg

    _skrl_spaces_pkg.unflatten_tensorized_space = unflatten_tensorized_space


_apply_unflatten_dict_patch()

from isaac_pursuit_evasion.skrl_ext.runner import CustomRunner  # noqa: E402

__all__ = ["CustomRunner"]
