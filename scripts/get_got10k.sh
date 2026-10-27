#!/usr/bin/env bash
# GOT-10k train: full_data.zip (70.7 GB) -> <train_data>/got10k/train (74 GB, 9 335 videos).
#   scripts/get_got10k.sh [Google Drive share link | URL | rclone source remote:folder | path of full_data.zip]
# Default source: full_data.zip on Hugging Face (GOT10K_HF: two parts, together the official full_data.zip of
# 70 692 770 032 bytes; no account needed; 16 connections with rclone). A Google Drive link is downloaded with
# rclone and your Google account when the remote gdrive is set up (scripts/gdrive.sh; Google often refuses anonymous
# downloads to cloud machines), otherwise directly. A downloaded archive is deleted after extraction (KEEP_ARCHIVES=1
# keeps it; needed for testing on GOT-10k val / test); an archive given as a path is never deleted.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_env
GOT10K_HF=${GOT10K_HF:-https://huggingface.co/datasets/jinyoungkim/GoT-10k/resolve/main/full_data.zip}
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
args=(--delete-archives)
if [ "${KEEP_ARCHIVES:-0}" = 1 ]; then args=(); fi
ARCHIVES=$(train_path archives)/got10k
READY=$(train_path train_data)/got10k/train/list.txt
have_archive() {  # a complete zip in the archive folder (its central directory is at the end)
  local z
  for z in "$ARCHIVES"/*.zip; do
    if [ -f "$z" ] && unzip -l "$z" >/dev/null 2>&1; then return 0; fi
  done
  return 1
}
hf_full_data() {  # <archives>/got10k/full_data.zip from the two parts on Hugging Face
  local zip=$ARCHIVES/full_data.zip p size have
  if [ -f "$READY" ] || have_archive; then rm -f "$zip".part??; return 0; fi
  mkdir -p "$ARCHIVES"
  for p in aa ab; do
    size=$(curl -fsSIL "$GOT10K_HF.part$p" | tr -d '\r' | sed -n 's/^content-length: *//Ip' | tail -1)
    have=$(stat -c %s "$zip.part$p" 2>/dev/null || echo 0)
    if [ "$have" -gt "$size" ]; then truncate -s "$size" "$zip.part$p"; have=$size; fi  # an interrupted joining
    if [ "$have" != "$size" ]; then
      echo "Downloading full_data.zip.part$p ($((size / 1000000000)) GB) from $GOT10K_HF.part$p"
      if command -v rclone >/dev/null; then  # 16 connections (one gets ~10 MB/s); starts the part again
        rclone copyto ":http:${GOT10K_HF##*/}.part$p" "$zip.part$p" --http-url "${GOT10K_HF%/*}/" \
          --multi-thread-streams 16 --multi-thread-cutoff 64M --retries 5 -P
      else
        curl -fL --retry 5 -C - -o "$zip.part$p" "$GOT10K_HF.part$p"
      fi
      if [ "$(stat -c %s "$zip.part$p")" != "$size" ]; then
        echo "full_data.zip.part$p is incomplete; run the script again" >&2; exit 1
      fi
    fi
  done
  echo "Joining the parts into $zip"
  cat "$zip.partab" >> "$zip.partaa" && mv "$zip.partaa" "$zip" && rm "$zip.partab"
}
if [ $# -eq 0 ]; then hf_full_data; fi
GDRIVE=""
if command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -qxF "gdrive:"; then GDRIVE=1; fi
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
