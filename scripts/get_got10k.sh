#!/usr/bin/env bash
# GOT-10k train: full_data.zip (70.7 GB) -> <train_data>/got10k/train (74 GB, 9 335 videos).
#   scripts/get_got10k.sh [rclone source remote:folder | Google Drive share link | URL | path of full_data.zip]
# Default source: GOT10K_RCLONE (a Google Drive folder with a copy of full_data.zip) if its rclone remote is set up
# (scripts/gdrive.sh), otherwise GOT10K_URL (the same copy, shared by link; Google may refuse anonymous downloads to
# cloud machines). A downloaded archive is deleted after extraction (KEEP_ARCHIVES=1 keeps it; needed for testing on
# GOT-10k val / test); an archive given as a path is never deleted.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_env
GOT10K_RCLONE=${GOT10K_RCLONE:-gdrive:LOKAP/train_archives/got10k}
GOT10K_URL=${GOT10K_URL:-https://drive.google.com/file/d/1JaRq_cBAR7p3JyZA4bL_TukbmrhUhdj-/view?usp=share_link}
is_rclone() {  # remote:path (not a URL, not an existing file)
  [[ $1 =~ ^[A-Za-z0-9_.-]+: && ! $1 =~ ^(https?|ftp):// && ! -e $1 ]]
}
if [ $# -eq 0 ]; then
  if command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -qxF "${GOT10K_RCLONE%%:*}:"; then
    set -- "$GOT10K_RCLONE"
  else
    set -- "$GOT10K_URL"
  fi
fi
args=(--delete-archives)
if [ "${KEEP_ARCHIVES:-0}" = 1 ]; then args=(); fi
ARCHIVES=$(train_path archives)/got10k
for src in "$@"; do
  if is_rclone "$src"; then
    # Into the archive folder, where prepare-train-data finds it; not again once the train split is extracted
    if [ ! -f "$(train_path train_data)/got10k/train/list.txt" ]; then
      echo "rclone: $src -> $ARCHIVES"
      rclone copy "$src" "$ARCHIVES" --include "*.{zip,tar,tar.gz,tgz}" --multi-thread-streams 8 -P
    fi
  else
    args+=(--got10k-url "$src")
  fi
done
stark prepare-train-data --datasets got10k "${args[@]}"
