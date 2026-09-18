#!/usr/bin/env bash
# install.sh — one-shot environment setup for MemoryStateCritic
#
# Requirements:
#   • Ubuntu 22.04 or 24.04
#   • NVIDIA GPU with driver ≥ 535 (CUDA 12.x)
#   • Python 3.11 available (conda, system, or installed by uv)
#   • Git with LFS
#   • Internet access (PyPI + pypi.nvidia.com)
#
# Usage:
#   chmod +x install.sh
#   ./install.sh          # installs into .venv/ inside the repo
#
# After installation activate with:
#   source .venv/bin/activate
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This must be the exact tree that produced the paper's runs, which is not an official
# Isaac Lab release: github.com/colson-louis/IsaacLab at d2579ea. That commit has since
# diverged from the fork's main, so cloning the fork and checking out the SHA is not
# reliable. It was submitted upstream as pull request #5725, and GitHub keeps PR head
# refs fetchable indefinitely, so we fetch it from the official repository.
# Building against any other commit silently changes the physics layer.
ISAACLAB_COMMIT="d2579eacf8eba0864e2328f3f383a0fdb411e00b"
ISAACLAB_FETCH_REF="refs/pull/5725/head"
ISAACLAB_DIR="$REPO_ROOT/IsaacLab"
VENV_DIR="$REPO_ROOT/.venv"

# ---------------------------------------------------------------------------
# 1. Install uv if missing
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "==> uv $(uv --version)"

# ---------------------------------------------------------------------------
# 2. Create virtual env with Python 3.11
# ---------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating .venv with Python 3.11..."
  uv venv "$VENV_DIR" --python 3.11
fi
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# 3. Core Python deps (torch CUDA 12.8, gymnasium, skrl fork, project pkg)
# ---------------------------------------------------------------------------
echo "==> Installing core Python dependencies..."
uv pip install \
  torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128

uv pip install \
  numpy==1.26.0 \
  gymnasium==1.2.1 \
  PyYAML==6.0.2 \
  toml==0.10.2 \
  psutil==5.9.8 \
  packaging==23.0 \
  tensorboard==2.18.0 \
  tqdm==4.67.1 \
  pytest==8.3.5 \
  pytest-mock==3.15.1 \
  omegaconf==2.3.0 \
  hydra-core==1.3.2 \
  wandb \
  tensordict \
  scipy==1.15.3 \
  matplotlib==3.10.3 \
  click==8.1.7 \
  starlette==0.45.3 \
  pandas \
  seaborn

# Isaac Lab's own dependencies. Step 6 installs Isaac Lab with --no-deps (its
# metadata pulls a conflicting torch), so they must be installed explicitly.
# Without these, `import isaaclab` fails on `warp` and no training run starts.
uv pip install \
  warp-lang==1.12.1 \
  prettytable==3.3.0 \
  trimesh \
  "pyglet<2" \
  "onnx>=1.18.0" \
  einops \
  transformers \
  pillow==11.3.0 \
  junitparser \
  flaky \
  hidapi==0.14.0.post2 \
  imageio imageio-ffmpeg \
  h5py \
  "protobuf>=4.25.8,!=5.26.0" \
  moviepy \
  rich \
  numba

# ---------------------------------------------------------------------------
# 4. Isaac Sim 5.1.0.0 from NVIDIA registry
# ---------------------------------------------------------------------------
echo "==> Installing Isaac Sim 5.1.0.0..."
uv pip install \
  isaacsim==5.1.0.0 \
  isaacsim-app==5.1.0.0 \
  isaacsim-core==5.1.0.0 \
  isaacsim-rl==5.1.0.0 \
  isaacsim-sensor==5.1.0.0 \
  isaacsim-robot==5.1.0.0 \
  isaacsim-utils==5.1.0.0 \
  isaacsim-asset==5.1.0.0 \
  isaacsim-extscache-kit==5.1.0.0 \
  isaacsim-extscache-kit-sdk==5.1.0.0 \
  isaacsim-extscache-physics==5.1.0.0 \
  isaacsim-replicator==5.1.0.0 \
  isaacsim-robot-motion==5.1.0.0 \
  isaacsim-robot-setup==5.1.0.0 \
  isaacsim-storage==5.1.0.0 \
  isaacsim-benchmark==5.1.0.0 \
  isaacsim-cortex==5.1.0.0 \
  isaacsim-example==5.1.0.0 \
  isaacsim-template==5.1.0.0 \
  isaacsim-test==5.1.0.0 \
  --extra-index-url https://pypi.nvidia.com

# Accept Isaac Sim EULA (non-interactive)
EULA_FILE="$VENV_DIR/lib/python3.11/site-packages/isaacsim/kit/EULA_ACCEPTED"
echo "yes" > "$EULA_FILE"
echo "==> Isaac Sim EULA accepted."

# ---------------------------------------------------------------------------
# 5. flatdict 4.0.1 — its setup.py uses pkg_resources which breaks in
#    isolated builds; fetch the sdist and patch it before installing.
# ---------------------------------------------------------------------------
echo "==> Installing flatdict 4.0.1 (with setup.py patch)..."
FLATDICT_TMP="$(mktemp -d)"
curl -sL "https://files.pythonhosted.org/packages/source/f/flatdict/flatdict-4.0.1.tar.gz" \
  -o "$FLATDICT_TMP/flatdict-4.0.1.tar.gz"
tar -xzf "$FLATDICT_TMP/flatdict-4.0.1.tar.gz" -C "$FLATDICT_TMP"
# Replace broken setup.py with a minimal one
cat > "$FLATDICT_TMP/flatdict-4.0.1/setup.py" <<'SETUP'
import setuptools
setuptools.setup()
SETUP
uv pip install "$FLATDICT_TMP/flatdict-4.0.1" --no-build-isolation
rm -rf "$FLATDICT_TMP"

# ---------------------------------------------------------------------------
# 6. Isaac Lab at pinned commit
# ---------------------------------------------------------------------------
if [ ! -d "$ISAACLAB_DIR/.git" ]; then
  echo "==> Cloning Isaac Lab @ $ISAACLAB_COMMIT..."
  git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR" --no-checkout
  git -C "$ISAACLAB_DIR" fetch --no-tags origin "$ISAACLAB_FETCH_REF"
  git -C "$ISAACLAB_DIR" checkout "$ISAACLAB_COMMIT"
  test "$(git -C "$ISAACLAB_DIR" rev-parse HEAD)" = "$ISAACLAB_COMMIT" \
    || { echo "ERROR: Isaac Lab is not at $ISAACLAB_COMMIT" >&2; exit 1; }
else
  echo "==> Isaac Lab already cloned at $ISAACLAB_DIR"
fi

echo "==> Installing Isaac Lab packages..."
for pkg in isaaclab isaaclab_rl isaaclab_tasks isaaclab_assets; do
  uv pip install -e "$ISAACLAB_DIR/source/$pkg" --no-build-isolation --no-deps
done

# ---------------------------------------------------------------------------
# 7. Bundled skrl fork and project package
# ---------------------------------------------------------------------------
echo "==> Installing bundled skrl fork and project package..."
uv pip install -e "$REPO_ROOT/source/third_parties/skrl"
uv pip install -e "$REPO_ROOT/source/isaac_pursuit_evasion"

# ---------------------------------------------------------------------------
# 8. Smoke tests
# ---------------------------------------------------------------------------
echo "==> Running smoke tests..."
# tests/conftest.py stubs Isaac Sim, so pytest passing does NOT prove the
# environment can train. Import the real stack first.
python - <<'SMOKE'
import importlib, sys
missing = [m for m in ("warp", "isaaclab", "pandas", "seaborn", "skrl", "torch")
           if not importlib.util.find_spec(m)]
if missing:
    sys.exit("ERROR: installed environment is missing: " + ", ".join(missing))
import isaaclab, torch  # noqa: F401  (isaaclab imports warp at module level)
print("  real-stack import OK (isaaclab, torch, plotting deps)")
SMOKE
python -m pytest "$REPO_ROOT/tests/" -q

echo ""
echo "==> Installation complete."
echo "    Activate the environment with:"
echo "        source .venv/bin/activate"
echo ""
echo "    Run training:"
echo "        python scripts/skrl/train.py --task=Pretrain-rl_rate-vs-trajectories --headless --num_envs=512"
echo ""
echo "    NOTE: Crazyflie USD assets are not in the repo."
echo "    Ask the repo owner for the Crazyflie/ directory and place it at:"
echo "        source/isaac_pursuit_evasion/assets/Crazyflie/"
