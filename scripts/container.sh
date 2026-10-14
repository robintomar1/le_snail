#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${ACCEPT_EULA:-}" != Y ]]; then
  printf '%s\n' 'Set ACCEPT_EULA=Y after reviewing NVIDIA Isaac Sim terms to launch the container.' >&2
  exit 2
fi
exec docker run --rm --gpus all --ipc=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT="${PRIVACY_CONSENT:-N}" \
  -v "$project_dir:/workspace/le_snail" \
  -v le-snail-kit-cache:/isaac-sim/kit/cache \
  -v le-snail-ov-cache:/root/.cache/ov \
  -w /workspace/le_snail le-snail:isaac-2.3.2 "$@"
