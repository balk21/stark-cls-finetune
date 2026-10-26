#!/usr/bin/env bash
# Starts `python -m stark_ft train <arguments>` in the tmux session "train", so it keeps running when SSH disconnects.
#   scripts/train.sh --set 'datasets=[got10k, coco]'
# The same arguments again resume the run from its last finished epoch. SESSION=<name> for another session name.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_env
# parameters, data, initial weights and run folder; stops on a problem
stark train "$@" --dry-run || { echo "Not started." >&2; exit 1; }
in_tmux "${SESSION:-train}" "$PY" -m stark_ft train "$@"
