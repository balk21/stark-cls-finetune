#!/usr/bin/env bash
# Sets up the rclone remote "gdrive" (Google Drive: GOT-10k, backups) from a token, without `rclone config`.
#   scripts/gdrive.sh '<token>'
# The token: `rclone authorize "drive"` on a computer with a browser (Mac: brew install rclone) prints it as one line,
# {"access_token": ...}. Running it again with a new token replaces the remote.
set -euo pipefail
if [ $# -ne 1 ]; then echo "usage: scripts/gdrive.sh '<the {...} line printed by rclone authorize \"drive\">'" >&2; exit 1; fi
command -v rclone >/dev/null || { echo "rclone is missing; run scripts/setup.sh first" >&2; exit 1; }
TOKEN=$(printf '%s' "$1" | tr -d '\r\n')
# Tried as a temporary remote first, so a wrong token does not replace a working remote
if ! RCLONE_CONFIG_GDRIVETEST_TYPE=drive RCLONE_CONFIG_GDRIVETEST_SCOPE=drive RCLONE_CONFIG_GDRIVETEST_TOKEN="$TOKEN" \
     rclone lsd gdrivetest: >/dev/null 2>&1; then
  echo "The token does not work; run rclone authorize \"drive\" again and paste the whole {...} line" >&2; exit 1
fi
if rclone listremotes 2>/dev/null | grep -qxF "gdrive:"; then rclone config delete gdrive; fi
CONF=$(rclone config file | tail -1)
mkdir -p "$(dirname "$CONF")"
printf '\n[gdrive]\ntype = drive\nscope = drive\ntoken = %s\n' "$TOKEN" >> "$CONF"
echo "Google Drive: the rclone remote gdrive is set up"
