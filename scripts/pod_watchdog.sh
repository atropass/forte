#!/usr/bin/env bash
# Restart a supervisord program when its health endpoint stops answering (a hung process does not crash, so
# supervisord's autorestart alone would never notice).
#   pod_watchdog.sh <supervisord.conf> <program> <health-url> [grace-seconds]
set -u
CONF="$1"; PROG="$2"; URL="$3"; GRACE="${4:-300}"
FAILS=0
sleep "$GRACE"
while true; do
  if curl -sf -m 10 "$URL" >/dev/null; then
    FAILS=0
  else
    FAILS=$((FAILS + 1))
    echo "$(date -Is) $PROG health check failed ($FAILS/3)"
  fi
  if [[ $FAILS -ge 3 ]]; then
    echo "$(date -Is) restarting $PROG"
    supervisorctl -c "$CONF" restart "$PROG"
    FAILS=0
    sleep "$GRACE"
  fi
  sleep 15
done
