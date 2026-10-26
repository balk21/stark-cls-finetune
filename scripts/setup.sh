#!/usr/bin/env bash
# Prepares a fresh GPU machine: system tools, the vot1 environment (micromamba, ~5-10 min), the paths, a check.
#   scripts/setup.sh [DATA_DIR]
# DATA_DIR: training data -> DATA_DIR/train, training runs -> DATA_DIR/training (configs/paths.local.yaml).
# Without it: inside the repository (data/train, outputs/training).
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
  { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux unzip curl bzip2 rclone >/dev/null; } ||
    echo "apt-get failed: install tmux, unzip, curl, bzip2 and rclone yourself" >&2
fi

if [ -z "$PY" ]; then
  echo "Creating the vot1 environment in $MAMBA_ROOT_PREFIX (~5-10 min) ..."
  MM=$MAMBA_ROOT_PREFIX/bin/micromamba
  if [ ! -x "$MM" ]; then
    mkdir -p "$MAMBA_ROOT_PREFIX"
    curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MAMBA_ROOT_PREFIX" bin/micromamba
  fi
  if [ -z "${CONDA_OVERRIDE_CUDA:-}" ]; then  # CUDA version of the driver, for the solver
    CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
    if [ -n "$CUDA" ]; then export CONDA_OVERRIDE_CUDA=$CUDA; fi
  fi
  "$MM" create -y -q -n vot1 -f "$REPO/environment/vot1_environment.yml"
  "$MM" clean -y -a -q  # package cache (~3 GB)
  PY=$MAMBA_ROOT_PREFIX/envs/vot1/bin/python
fi

if [ $# -ge 1 ]; then
  DATA=$(realpath -m "$1")
  mkdir -p "$DATA"
  printf '# Written by scripts/setup.sh\ntrain_data: %s/train\ntrain_outputs: %s/training\n' "$DATA" "$DATA" \
    > "$REPO/configs/paths.local.yaml"
fi

echo "vot1 python: $PY"
stark check
