#!/usr/bin/env bash
# LaSOT: Hugging Face l-lt/LaSOT (70 class zips, 248 GB, all 1 400 videos) -> <train_data>/lasot/<class>/<class>-<n>/.
# Six downloads in parallel; every zip is extracted, then deleted. Running it again continues where it stopped.
# Training uses the 1 120 videos of lib/train/data_specs/lasot_train_split.txt (checked at the end).
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_env
HF=${LASOT_HF:-https://huggingface.co}
DEST=$(train_path train_data)/lasot
ZIPS=$(train_path archives)/lasot  # zips while downloading / extracting, and <class>.done markers
mkdir -p "$DEST" "$ZIPS"

# Classes still to do; stops if the free disk space is not enough for them
CLASSES=$(curl -fsSL "$HF/api/datasets/l-lt/LaSOT/tree/main" | "$PY" -c '
import json, os, shutil, sys
zips, dest = sys.argv[1], sys.argv[2]
todo = [(f["path"][:-4], f.get("lfs", {}).get("size", f.get("size", 0))) for f in json.load(sys.stdin)
        if f["path"].endswith(".zip") and not os.path.exists(os.path.join(zips, f["path"][:-4] + ".done"))]
need = sum(s for _, s in todo) + 6 * max([s for _, s in todo] or [0])  # extracted + zips in progress
free = shutil.disk_usage(dest).free
if need > free:
    sys.exit(f"Not enough disk space for LaSOT: {need / 1e9:.0f} GB needed, {free / 1e9:.0f} GB free in {dest}")
print("\n".join(c for c, _ in todo))' "$ZIPS" "$DEST")

one() {  # one <class>: download, extract into <class>.tmp, rename, delete the zip
  set -euo pipefail
  local c=$1 zip=$ZIPS/$1.zip
  if ! { [ -f "$zip" ] && unzip -tqq "$zip" >/dev/null 2>&1; }; then
    curl -fsSL --retry 5 -C - -o "$zip" "$HF/datasets/l-lt/LaSOT/resolve/main/$c.zip"
  fi
  rm -rf "$DEST/$c.tmp" && unzip -q "$zip" -d "$DEST/$c.tmp"
  rm -rf "${DEST:?}/$c" && mv "$DEST/$c.tmp" "$DEST/$c" && touch "$ZIPS/$c.done" && rm -f "$zip"
  echo "$c done"
}
export -f one
export HF DEST ZIPS
if [ -n "$CLASSES" ]; then
  echo "LaSOT: $(echo "$CLASSES" | wc -l) classes to download into $DEST"
  echo "$CLASSES" | xargs -P 6 -I{} bash -c 'one "$1"' _ {}
fi

(cd "$REPO" && "$PY" -c '
import sys
from pathlib import Path
root = Path(sys.argv[1])
seqs = [s.strip() for s in open("lib/train/data_specs/lasot_train_split.txt") if s.strip()]
missing = [s for s in seqs if not (root / s.split("-")[0] / s / "groundtruth.txt").is_file()]
print(f"LaSOT training videos: {len(seqs) - len(missing)} / {len(seqs)}" + (f", missing e.g. {missing[:5]}" if missing else ""))
sys.exit(1 if missing else 0)' "$DEST")
