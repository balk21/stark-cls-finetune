#!/usr/bin/env bash
# GOT-10k train: full_data.zip (70.7 GB) -> <train_data>/got10k/train (74 GB, 9 335 videos).
#   scripts/get_got10k.sh <Google Drive share link | URL | path of full_data.zip>
# A downloaded archive is deleted after extraction (KEEP_ARCHIVES=1 keeps it; needed for testing on GOT-10k val /
# test); an archive given as a path is never deleted.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_env
if [ $# -lt 1 ]; then echo "usage: scripts/get_got10k.sh <Google Drive share link | URL | path of full_data.zip>" >&2; exit 1; fi
args=(--delete-archives)
if [ "${KEEP_ARCHIVES:-0}" = 1 ]; then args=(); fi
for src in "$@"; do args+=(--got10k-url "$src"); done
stark prepare-train-data --datasets got10k "${args[@]}"
