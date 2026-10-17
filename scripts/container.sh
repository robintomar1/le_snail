#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${ACCEPT_EULA:-}" != Y ]]; then
  printf '%s\n' 'Set ACCEPT_EULA=Y after reviewing NVIDIA Isaac Sim terms to launch the container.' >&2
  exit 2
fi
display_args=()
if [[ -n "${LE_SNAIL_DISPLAY:-}" ]]; then
  # GUI (scripts/play.py): the host X display and its auth cookie.
  : "${DISPLAY:?LE_SNAIL_DISPLAY needs DISPLAY}"
  xauth_file="${XAUTHORITY:-$HOME/.Xauthority}"
  # Same hostname as the host so the local X cookie matches.
  display_args=(--hostname "$(hostname)" -e DISPLAY="$DISPLAY" -e XAUTHORITY=/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v "$xauth_file:/root/.Xauthority:ro")
fi
exec docker run --rm --gpus all --ipc=host "${display_args[@]}" \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT="${PRIVACY_CONSENT:-N}" \
  -v "$project_dir:/workspace/le_snail" \
  -v le-snail-kit-cache:/isaac-sim/kit/cache \
  -v le-snail-ov-cache:/root/.cache/ov \
  -w /workspace/le_snail le-snail:isaac-2.3.2 "$@"
