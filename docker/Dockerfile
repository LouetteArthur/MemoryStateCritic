# Base: CUDA 12.8 development image (matches torch==2.7.0+cu128)
# Isaac Sim 5.1.0.0 is installed via pip from the NVIDIA registry — no NGC
# monolithic image needed, which eliminates the old /isaac-sim path confusion.
ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

# ── Build-time environment ────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    # Allow Isaac Sim / OmniKit to run as root (required in Docker containers)
    OMNI_KIT_ALLOW_ROOT=1 \
    # Accept Isaac Sim EULA and telemetry opt-out non-interactively
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    # Suppress pip root-user warning
    PIP_ROOT_USER_ACTION=ignore

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git curl ca-certificates \
    # Runtime libs needed by Isaac Sim / PhysX / Vulkan
    libglib2.0-0 libsm6 libxext6 libxrender1 libx11-dev libxt6 \
    libgl1 libgles2 libegl1 libvulkan1 libglu1-mesa \
    libgomp1 libnuma1 \
    # Audio stub (Isaac Sim expects PulseAudio presence; this satisfies the check)
    libpulse0 \
    && rm -rf /var/lib/apt/lists/*

# ── NVIDIA Vulkan ICD (needed for headless GPU rendering in Docker) ───────────
RUN mkdir -p /usr/share/vulkan/icd.d \
    && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}\n' \
         > /usr/share/vulkan/icd.d/nvidia_icd.json
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Make python3.11 the default python
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel

# ── PyTorch 2.7.0 + CUDA 12.8 ────────────────────────────────────────────────
RUN python3 -m pip install --no-cache-dir \
    torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# ── Isaac Sim 5.1.0.0 (pip, NVIDIA registry) ─────────────────────────────────
# Split into two RUN layers so Docker cache is more granular.
RUN python3 -m pip install --no-cache-dir \
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
    --extra-index-url https://pypi.nvidia.com

RUN python3 -m pip install --no-cache-dir \
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

# Accept Isaac Sim EULA via the file mechanism (matches install.sh exactly).
# Use pip show to locate the package — importing isaacsim itself triggers an
# interactive EULA prompt that blocks the non-interactive build.
RUN SITE=$(python3 -m pip show isaacsim | grep ^Location | awk '{print $2}') \
    && mkdir -p "$SITE/isaacsim/kit" \
    && echo "yes" > "$SITE/isaacsim/kit/EULA_ACCEPTED" \
    && echo "[Docker] Isaac Sim EULA accepted at $SITE/isaacsim/kit/EULA_ACCEPTED"

# ── Isaac Lab at pinned commit ────────────────────────────────────────────────
# This must be the *exact* tree that produced the paper's runs, which is not an
# official Isaac Lab release: the host used github.com/colson-louis/IsaacLab at
# d2579ea. That commit has since diverged from the fork's main (14 ahead / 4
# behind), so cloning the fork and checking out the SHA is not reliable. It was
# submitted upstream as PR #5725, and GitHub keeps pull-request head refs
# fetchable from the official repository indefinitely -- so we fetch it there.
#
# Building with any other commit silently changes the physics layer and the runs
# are no longer poolable with the existing ones.
ARG ISAACLAB_COMMIT=d2579eacf8eba0864e2328f3f383a0fdb411e00b
ARG ISAACLAB_FETCH_REF=refs/pull/5725/head
RUN git clone https://github.com/isaac-sim/IsaacLab.git /IsaacLab \
    && git -C /IsaacLab fetch --no-tags origin ${ISAACLAB_FETCH_REF} \
    && git -C /IsaacLab checkout ${ISAACLAB_COMMIT} \
    && test "$(git -C /IsaacLab rev-parse HEAD)" = "${ISAACLAB_COMMIT}" \
    && echo "[Docker] Isaac Lab pinned at ${ISAACLAB_COMMIT}"

RUN for pkg in isaaclab isaaclab_rl isaaclab_tasks isaaclab_assets; do \
      python3 -m pip install --no-build-isolation --no-deps \
        -e /IsaacLab/source/$pkg; \
    done

# ── General Python dependencies ───────────────────────────────────────────────
RUN python3 -m pip install --no-cache-dir \
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
    "hydra-core==1.3.2" \
    wandb \
    tensordict \
    "scipy==1.15.3" \
    "matplotlib==3.10.3" \
    "click==8.1.7" \
    "starlette==0.45.3" \
    imageio imageio-ffmpeg \
    "warp-lang==1.12.1" \
    # Isaac Lab deps (skipped by --no-deps above, listed in isaaclab/setup.py)
    "prettytable==3.3.0" \
    trimesh \
    "pyglet<2" \
    "onnx>=1.18.0" \
    einops \
    transformers \
    "pillow==11.3.0" \
    junitparser \
    flaky \
    "hidapi==0.14.0.post2" \
    # Isaac Lab RL / tasks deps
    h5py \
    "protobuf>=4.25.8,!=5.26.0" \
    moviepy \
    rich \
    numba \
    # figure regeneration (scripts/plot_ablation_results.py)
    pandas \
    seaborn

# ── flatdict 4.0.1 (setup.py uses pkg_resources — must patch before install) ──
RUN curl -sL "https://files.pythonhosted.org/packages/source/f/flatdict/flatdict-4.0.1.tar.gz" \
      -o /tmp/flatdict.tar.gz \
    && mkdir -p /tmp/flatdict \
    && tar -xzf /tmp/flatdict.tar.gz -C /tmp/flatdict \
    && printf 'import setuptools\nsetuptools.setup()\n' \
         > /tmp/flatdict/flatdict-4.0.1/setup.py \
    && python3 -m pip install /tmp/flatdict/flatdict-4.0.1 --no-build-isolation \
    && rm -rf /tmp/flatdict /tmp/flatdict.tar.gz

# ── Project entrypoint ────────────────────────────────────────────────────────
WORKDIR /workspace/IsaacPursuitEvasion

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
