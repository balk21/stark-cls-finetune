#!/usr/bin/env bash
# `python -m stark_ft <arguments>` in the vot1 environment, without activating it.
#   scripts/stark.sh train-report st101_got10k_stage1
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_env
stark "$@"
