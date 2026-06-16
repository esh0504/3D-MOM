#!/usr/bin/env bash
set -euo pipefail

XVFB_DISPLAY="${XVFB_DISPLAY:-:99}"
XVFB_RESOLUTION="${XVFB_RESOLUTION:-1920x1080x24}"

if [ -z "${DISPLAY:-}" ]; then
  export DISPLAY="${XVFB_DISPLAY}"
fi

if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
  echo "[entrypoint] Starting Xvfb on ${DISPLAY} (${XVFB_RESOLUTION})"
  Xvfb "${DISPLAY}" -screen 0 "${XVFB_RESOLUTION}" -ac +extension GLX +render -noreset &
  xvfb_pid=$!
  for _ in $(seq 1 20); do
    if xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
      echo "[entrypoint] Xvfb ready (pid ${xvfb_pid})"
      break
    fi
    sleep 0.25
  done
  if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    echo "[entrypoint] Failed to start Xvfb on ${DISPLAY}" >&2
    exit 1
  fi
fi

exec "$@"
