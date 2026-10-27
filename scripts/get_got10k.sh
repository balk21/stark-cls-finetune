#!/usr/bin/env bash
# GOT-10k train: full_data.zip (70.7 GB) -> <train_data>/got10k/train (74 GB, 9 335 videos).
#   scripts/get_got10k.sh [Google Drive share link | URL | rclone source remote:folder | path of full_data.zip]
# Default source: GOT10K_URL (a copy of full_data.zip on Google Drive, shared by link). A Google Drive link is
# downloaded with rclone and your Google account when the remote gdrive is set up (scripts/gdrive.sh; Google often
# refuses anonymous downloads to cloud machines), otherwise directly. A downloaded archive is deleted after extraction
# (KEEP_ARCHIVES=1 keeps it; needed for testing on GOT-10k val / test); an archive given as a path is never deleted.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_env
GOT10K_URL=${GOT10K_URL:-https://drive.google.com/file/d/1JaRq_cBAR7p3JyZA4bL_TukbmrhUhdj-/view?usp=share_link}
is_rclone() {  # remote:path (not a URL, not an existing file)
  [[ $1 =~ ^[A-Za-z0-9_.-]+: && ! $1 =~ ^(https?|ftp):// && ! -e $1 ]]
}
drive_id() {  # the file ID of a Google Drive link; nothing for other sources
  if [[ $1 =~ (drive\.google\.com/(file/d/|open\?([^#]*&)?id=|uc\?([^#]*&)?id=)|drive\.usercontent\.google\.com/download\?([^#]*&)?id=)([A-Za-z0-9_-]{20,}) ]]; then
    echo "${BASH_REMATCH[6]}"
  fi
}
retry() {  # rclone <arguments>, again after a minute (short rate limits of rclone's shared Google client)
  local try rc
  for try in 1 2 3 4 5; do
    if rclone "$@"; then return 0; else rc=$?; fi
    if [ "$rc" = 3 ] || [ "$rc" = 4 ] || [ "$try" = 5 ]; then return "$rc"; fi  # 3, 4: not found
    echo "rclone failed; trying again in 1 minute ($try / 5)" >&2
    sleep 60
  done
}
if [ $# -eq 0 ]; then set -- "$GOT10K_URL"; fi
args=(--delete-archives)
if [ "${KEEP_ARCHIVES:-0}" = 1 ]; then args=(); fi
ARCHIVES=$(train_path archives)/got10k
READY=$(train_path train_data)/got10k/train/list.txt
GDRIVE=""
if command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -qxF "gdrive:"; then GDRIVE=1; fi
have_archive() {  # a complete zip in the archive folder (its central directory is at the end)
  local z
  for z in "$ARCHIVES"/*.zip; do
    if [ -f "$z" ] && unzip -l "$z" >/dev/null 2>&1; then return 0; fi
  done
  return 1
}
for src in "$@"; do
  id=$(drive_id "$src")
  if [ -n "$id" ] && [ -n "$GDRIVE" ] || is_rclone "$src"; then
    # Into the archive folder, where prepare-train-data finds it; not again once it is there or extracted
    if [ -f "$READY" ] || have_archive; then continue; fi
    mkdir -p "$ARCHIVES"
    if [ -n "$id" ]; then
      echo "rclone: Google Drive file $id -> $ARCHIVES"
      retry backend copyid gdrive: "$id" "$ARCHIVES/" --multi-thread-streams 8 -P || exit 1
    else
      echo "rclone: $src -> $ARCHIVES"
      retry copy "$src" "$ARCHIVES" --include "*.{zip,tar,tar.gz,tgz}" --multi-thread-streams 8 -P || exit 1
    fi
  else
    args+=(--got10k-url "$src")
  fi
done
stark prepare-train-data --datasets got10k "${args[@]}"
