#!/usr/bin/env bash
# Start the CV service (RT-DETRv2) + nginx gateway on the CV GPU pod under supervisord.
# Secrets come from /workspace/secrets.env (chmod 600, never committed), e.g.:
#     CV_API_KEY=<long random token>
set -euo pipefail

SECRETS="${SECRETS_FILE:-/workspace/secrets.env}"
CONF=/workspace/forte/configs/supervisord.cv.conf

echo "==> [ForteVision] Starting the CV service"
if [[ ! -f "$SECRETS" ]]; then
  echo "error: $SECRETS not found (needs CV_API_KEY=...)" >&2
  exit 1
fi
chmod 600 "$SECRETS"
set -a
# shellcheck disable=SC1090
source "$SECRETS"
set +a
if [[ -z "${CV_API_KEY:-}" || ${#CV_API_KEY} -lt 24 ]]; then
  echo "error: CV_API_KEY missing or shorter than 24 characters in $SECRETS" >&2
  exit 1
fi

# stop only the distro nginx service, so our gateway can bind :8000 (do not kill other nginx instances)
service nginx stop 2>/dev/null || true

if ! pgrep -f "supervisord -c $CONF" > /dev/null; then
  /usr/bin/supervisord -c "$CONF"
  echo "==> supervisord started"
else
  echo "==> supervisord already running: restarting programs with the current secrets"
  supervisorctl -c "$CONF" shutdown || true
  sleep 3
  /usr/bin/supervisord -c "$CONF"
fi

for _ in $(seq 1 200); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then
    echo "==> CV service healthy"
    exit 0
  fi
  sleep 3
done
echo "error: CV service did not become healthy; see /workspace/logs/cv_service_err.log" >&2
exit 1
