#!/usr/bin/env bash
# Copies a training run from cloud storage (e.g. a backup) to <train_outputs>/<run name>/, to resume it here.
#   scripts/restore.sh <run name> <remote:folder>                e.g. scripts/restore.sh st101_got10k_stage1 gdrive:LOKAP/training
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_env
if [ $# -lt 2 ]; then echo "usage: scripts/restore.sh <run name> <remote:folder>" >&2; exit 1; fi
RUN_DIR=$(train_path train_outputs)/$1
rclone copy "$2/$1" "$RUN_DIR" --progress
ls -la "$RUN_DIR" "$RUN_DIR/checkpoints"
