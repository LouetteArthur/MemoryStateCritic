#!/bin/bash
set -e

# CRITICAL: use `python` (not `python3`) here. In images where Isaac Sim 5.1
# is installed as a pip package, the install diverges /usr/local/bin/python
# (→ Isaac Sim's embedded /isaac-sim/kit/python) from /usr/local/bin/python3
# (→ system /usr/bin/python3.11). Train scripts call `python`, so we MUST
# install editable packages into that interpreter's site-packages or the
# train will fall back to the upstream pip-installed skrl whose obs routing
# breaks our PPO_RNN_VSH/SZZ/SHH critics with a (8196 vs 64) tensor shape
# mismatch in the critic_state_preprocessor.
PY="$(command -v python)"
echo "[entrypoint] Using python: $PY"

$PY -m pip install -q --no-build-isolation --no-deps --force-reinstall \
    -e /workspace/MemoryStateCritic/source/third_parties/skrl 2>&1 || \
    echo "[entrypoint] WARNING: failed to install local skrl"
$PY -m pip install -q --no-build-isolation --no-deps \
    -e /workspace/MemoryStateCritic/source/isaac_pursuit_evasion 2>&1 || \
    echo "[entrypoint] WARNING: failed to install isaac_pursuit_evasion"

# Sanity check: print the actual skrl path the train will use
$PY -c "import skrl; print('[entrypoint] skrl resolves to:', skrl.__file__)" 2>&1 || true

# Wire up the repo root so `import isaac_pursuit_evasion` and peer imports work.
export PYTHONPATH="/workspace/MemoryStateCritic:${PYTHONPATH}"

exec "$@"
