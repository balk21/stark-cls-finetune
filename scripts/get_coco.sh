#!/usr/bin/env bash
# COCO 2017 train: downloaded from images.cocodataset.org (19 GB) -> <train_data>/coco (20 GB).
# The archives are deleted after extraction (KEEP_ARCHIVES=1 keeps them).
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_env
args=(--delete-archives)
if [ "${KEEP_ARCHIVES:-0}" = 1 ]; then args=(); fi
stark prepare-train-data --datasets coco "${args[@]}"
