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
    if [ "$(tmux display-message -p -t "$session" '#{pane_dead}')" != 1 ]; then
      echo "tmux session '$session' is still running: tmux attach -t $session" >&2; exit 1
    fi
    echo "Closing the finished tmux session '$session' (exit status $(tmux display-message -p -t "$session" '#{pane_dead_status}'))"
    tmux kill-session -t "$session"
  fi
  # The pane stays open after the command ends (finished or crashed), with its output; the next start replaces it
  tmux new-session -d -s "$session" "sleep infinity"  # placeholder: remain-on-exit is set before the command starts
  tmux set-option -t "$session" remain-on-exit on >/dev/null
  tmux respawn-pane -k -t "$session" "cd $(printf %q "$REPO") && $(printf '%q ' "$@")"
  echo "Started in tmux session '$session': tmux attach -t $session   (leave it running: Ctrl-b d)"
}
