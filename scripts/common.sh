# Shared by the scripts in this folder (sourced, not run): repository root, the python of the vot1 environment and
# helpers.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}

_vot1_python() {  # VOT1_PYTHON, or the vot1 environment of micromamba / conda
  local p
  for p in "${VOT1_PYTHON:-}" "$MAMBA_ROOT_PREFIX/envs/vot1/bin/python" \
           "$HOME"/{miniforge3,miniconda3,anaconda3,mambaforge}/envs/vot1/bin/python /opt/conda/envs/vot1/bin/python; do
    if [ -n "$p" ] && [ -x "$p" ]; then echo "$p"; return 0; fi
  done
  return 1
}
PY=$(_vot1_python || true)

require_env() {
  [ -n "$PY" ] || { echo "The vot1 environment was not found; run scripts/setup.sh first." >&2; exit 1; }
}

stark() {  # python -m stark_ft ... in the vot1 environment
  (cd "$REPO" && "$PY" -m stark_ft "$@")
}

train_path() {  # train_data | train_outputs | archives (configs/paths*.yaml)
  (cd "$REPO" && "$PY" -c "from stark_ft.paths import get_paths; print(get_paths().$1)")
}

in_tmux() {  # in_tmux <session> <command ...>: runs the command in a detached tmux session (survives SSH)
  local session=$1; shift
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session '$session' already exists: tmux attach -t $session" >&2; exit 1
  fi
  tmux new-session -d -s "$session" "cd $(printf %q "$REPO") && $(printf '%q ' "$@"); echo; echo '[finished]'; exec bash"
  echo "Started in tmux session '$session': tmux attach -t $session   (leave it running: Ctrl-b d)"
}
