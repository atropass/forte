#!/usr/bin/env bash
# Restart a supervisord program when its health endpoint stops answering (a hung process does not crash, so
# supervisord's autorestart alone would never notice).
#   pod_watchdog.sh <supervisord.conf> <program> <health-url> [startup-timeout-seconds]
# A starting program is never interrupted: model loading from a network volume can take minutes. Only a program
# that has been healthy and then fails 3 checks in a row (~45 s) is restarted, or one that does not become healthy
# within the startup timeout (default 900 s).
set -u
CONF="$1"; PROG="$2"; URL="$3"; STARTUP="${4:-900}"
log() { echo "$(date -Is) $PROG: $*"; }
healthy() { curl -sf -m 10 "$URL" >/dev/null; }

while true; do
  # starting: wait until healthy (no restarts unless the startup timeout passes)
  t0=$(date +%s)
  until healthy; do
    if (( $(date +%s) - t0 > STARTUP )); then
      log "not healthy after ${STARTUP}s of startup: restarting"
      supervisorctl -c "$CONF" restart "$PROG"
      t0=$(date +%s)
    fi
    sleep 15
  done
  log "healthy"
  # running: restart after 3 consecutive failed checks
  fails=0
  while (( fails < 3 )); do
    sleep 15
    if healthy; then fails=0; else fails=$((fails + 1)); log "health check failed ($fails/3)"; fi
  done
  log "restarting after 3 failed health checks"
  supervisorctl -c "$CONF" restart "$PROG"
done
