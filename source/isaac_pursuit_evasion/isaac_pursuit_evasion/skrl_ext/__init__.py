"""Project-local extensions to skrl.

This package contains all the custom algorithmic additions used by
isaac_pursuit_evasion that are not part of upstream skrl.  Keeping them
here (rather than patching the vendored skrl copy) makes it trivial to:

- diff exactly what is original contribution vs library code
- upgrade skrl without merge conflicts
- share the project privately without shipping a forked library

Contents
--------
- ``models.gaussian_cnn_rnn``  : CNN+GRU actor with past-action concat
  (R2D2/IMPALA recipe), used for POMDPs with partial visual observability.
- ``models.vsh_critic``        : V(s, h) critic that concatenates the
  preprocessed privileged state with the detached actor GRU hidden state.
- ``agents.ppo_asym``          : PPO with asymmetric actor-critic support
  (critic operates on a separate privileged state space, optionally
  Dict-shaped for unbiased critics).
- ``agents.ppo_rnn_vsh``       : PPO_RNN with asymmetric critic support and
  optional V(s, h) mode (stores and passes actor GRU hidden state).
- ``runner``                   : ``CustomRunner`` that registers the
  models and agents above via the standard skrl YAML class names.

Import-time side effect
-----------------------
Importing this package monkey-patches
``skrl.utils.spaces.torch.unflatten_tensorized_space`` so it also accepts an
already-unflattened ``dict`` input, which is needed by the dict-shaped
unbiased critic path in ``PPO_ASYM``.  Upstream skrl only handles flat
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
