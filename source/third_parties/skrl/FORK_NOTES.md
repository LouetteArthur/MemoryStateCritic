# Vendored skrl fork — what differs from upstream 1.4.3

This directory is a fork of [skrl](https://github.com/Toni-SM/skrl) 1.4.3 by Toni-SM,
redistributed under its original MIT licence (see `LICENSE`, kept intact).

`skrl_1.4.3.patch` is the exact delta, generated with:

```bash
diff -ruN skrl-1.4.3/ source/third_parties/skrl/
```

It touches **7 files** (+485 / −64 functional lines). Nothing else in the package is
modified.

## Why a fork rather than a subclass

Upstream skrl 1.4.3 has no notion of a *privileged* critic input distinct from the actor's
observation. Asymmetric actor-critic needs the state `s` to reach the critic through the
rollout memory and to be normalised by its own running scaler. That plumbing crosses the
agent, the memory and the environment wrapper, which is more than a subclass can reach.
Changes that *could* live outside the fork do: see `source/isaac_pursuit_evasion/
isaac_pursuit_evasion/skrl_ext/`, which holds the models, the agent and one documented
monkey-patch rather than editing the vendored copy.

## Functional changes

| File | Change |
|---|---|
| `skrl/agents/torch/ppo/ppo.py` | Adds `critic_state_preprocessor` and `critic_state_preprocessor_kwargs` to the default config, so privileged states get their own `RunningStandardScaler` instead of reusing the actor's. |
| `skrl/agents/torch/ppo/ppo_rnn.py` | Same two config keys, plus: a `critic_states` rollout tensor sized from `critic_state_preprocessor_kwargs["size"]`; extraction of `critic_states` / `next_critic_states` from the env `info` dict; the preprocessor is registered in `checkpoint_modules` so it is saved and restored. Falls back to the actor's observation space when no critic preprocessor is configured. |
| `skrl/envs/wrappers/torch/isaaclab_envs.py` | The wrapper reads `observations["critic"]` and publishes it as `info["critic_states"]` and `info["next_critic_states"]`. **Alignment convention:** `info["critic_states"]` carries the state from *before* the transition (`prev_critic_states`), so the stored transition pairs `s_t` with `a_t`; `next_critic_states` carries `s_{t+1}` for bootstrapping. Getting this backwards shifts the critic by one step and is silent — `tests/test_gru_actor.py` guards the analogous property on the actor side. Also adds a `get_critic_states()` accessor. |
| `skrl/utils/model_instantiators/torch/gaussian_rnn.py` | **New file** (+249). Gaussian recurrent model instantiator, absent upstream; needed to build the CNN+GRU actor from YAML. |
| `skrl/utils/model_instantiators/torch/__init__.py` | Exports `gaussian_rnn_model`. |
| `skrl/utils/runner/torch/runner.py` | Registers the `gaussianrnnmixin` component name so YAML configs can select the new model. |
| `skrl/utils/spaces/torch/spaces.py` | Type-hint modernisation only (see below); no behavioural change. |

## A warning about reading the patch

The repository's `pre-commit` configuration was, at one point, run without excluding
`source/third_parties/`. As a result the patch also contains a large amount of **cosmetic
churn** from `pyupgrade --py310-plus` and `isort`: `Optional[X]` rewritten as `X | None`,
`typing.Mapping` moved to `collections.abc.Mapping`, imports reordered, signatures rewrapped.
None of it changes behaviour, but it inflates the diff well beyond the functional delta.

`spaces.py` is *entirely* this kind of churn and could be reverted to upstream with no effect.

The hooks now carry `exclude: "^source/third_parties/"` so this does not recur. If you
regenerate the patch, expect it to shrink if you first restore the cosmetically-touched
files from upstream.
