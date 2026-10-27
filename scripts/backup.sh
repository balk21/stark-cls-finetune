#!/usr/bin/env bash
# Copies a training run to cloud storage with rclone every 30 minutes, in the tmux session "backup-<run>".
#   scripts/backup.sh <run name> <remote:folder> [minutes]      e.g. scripts/backup.sh st101_got10k_stage1 gdrive:LOKAP/training
# The remote gdrive is set up with scripts/gdrive.sh (docs/server.md). Only changed files are copied.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_env
if [ $# -lt 2 ]; then echo "usage: scripts/backup.sh <run name> <remote:folder> [minutes]" >&2; exit 1; fi
RUN_DIR=$(train_path train_outputs)/$1
[ -d "$RUN_DIR" ] || { echo "No run folder $RUN_DIR" >&2; exit 1; }
in_tmux "backup-$1" bash -c 'while true; do date; rclone copy "$1" "$2" --exclude "tensorboard/**" && echo copied; sleep $(($3 * 60)); done' \
  _ "$RUN_DIR" "$2/$1" "${3:-30}"
