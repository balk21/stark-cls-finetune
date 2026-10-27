#!/usr/bin/env bash
# GOT-10k train: full_data.zip (70.7 GB) -> <train_data>/got10k/train (74 GB, 9 335 videos).
#   scripts/get_got10k.sh [Google Drive share link | URL | rclone source remote:folder | path of full_data.zip]
# Default source: full_data.zip on Hugging Face (GOT10K_HF: two parts, together the official full_data.zip of
# 70 692 770 032 bytes; no account needed; 16 connections). A Google Drive link is downloaded with
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
CONNECTIONS=16  # Hugging Face gives one connection ~10 MB/s
range() {  # range <url> <first byte> <last byte> <file>: continues a partly downloaded byte range
  local try have
  for try in 1 2 3 4 5 6; do  # each try continues after the bytes already in the file
    have=$(stat -c %s "$4" 2>/dev/null || echo 0)
    if [ $(($2 + have)) -gt "$3" ]; then return 0; fi
    curl -fsSL -r "$(($2 + have))-$3" "$1" >> "$4" || sleep 5
  done
  return 1
}
export -f range
hf_full_data() {  # <archives>/got10k/full_data.zip: the two parts on Hugging Face in byte ranges, then joined
  local zip=$ARCHIVES/full_data.zip jobs=$ARCHIVES/ranges.txt p i size step first last k=0 bytes=0
  if [ -f "$READY" ] || have_archive; then rm -f "$zip".a[ab][0-9][0-9] "$zip.joining" "$jobs"; return 0; fi
  mkdir -p "$ARCHIVES"
  rm -f "$zip".part??  # whole parts of an earlier version of this script
  for p in aa ab; do  # url, first byte, last byte, file: in the order of the joined file
    size=$(curl -fsSIL "$GOT10K_HF.part$p" | tr -d '\r' | sed -n 's/^content-length: *//Ip' | tail -1)
    step=$(((size + CONNECTIONS - 1) / CONNECTIONS))
    for ((i = 0; i < CONNECTIONS; i++)); do
      first=$((i * step)); last=$((first + step - 1)); if [ "$last" -ge "$size" ]; then last=$((size - 1)); fi
      printf '%s %d %d %s\n' "$GOT10K_HF.part$p" "$first" "$last" "$zip.$p$(printf %02d "$i")"
    done
  done > "$jobs"
  if [ -f "$zip.joining" ]; then  # an interrupted joining: ranges whose file is gone are in it
    while read -r _ first last f; do
      if [ -f "$f" ]; then break; fi
      k=$((k + 1)); bytes=$((bytes + last - first + 1))
    done < "$jobs"
    truncate -s "$bytes" "$zip.joining"
  fi
  echo "Downloading full_data.zip (70.7 GB, $CONNECTIONS connections) from ${GOT10K_HF%/*}"
  (while sleep 30; do
    stat -c %s "$zip".a[ab][0-9][0-9] 2>/dev/null | awk -v b="$bytes" '{b += $1} END {printf "  %.1f GB\n", b / 1e9}'
  done) &
  MONITOR=$!
  trap 'kill "$MONITOR" 2>/dev/null || true' EXIT
  tail -n +$((k + 1)) "$jobs" | xargs -P "$CONNECTIONS" -L 1 bash -c 'range "$@"' _
  kill "$MONITOR"
  echo "Joining the parts into $zip"
  tail -n +$((k + 1)) "$jobs" | while read -r _ first last f; do
    if [ "$(stat -c %s "$f")" != $((last - first + 1)) ]; then echo "$f is incomplete; run again" >&2; exit 1; fi
    cat "$f" >> "$zip.joining" && rm "$f"
  done
  mv "$zip.joining" "$zip" && rm "$jobs"
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
