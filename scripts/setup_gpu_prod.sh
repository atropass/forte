#!/usr/bin/env bash
# Production Setup & Supervisor Starter for vLLM on RunPod
set -euo pipefail

echo "==> Configuring Production Environment..."

# Install required system packages
DEBIAN_FRONTEND=noninteractive apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  nginx supervisor ninja-build curl procps

mkdir -p /workspace/logs /workspace/hf_home /workspace/.cache/vllm /workspace/.cache/torch_extensions

# Kill any existing standalone instances
pkill -f "vllm.entrypoints" || true
pkill -f "supervisord" || true
pkill -f "nginx: master" || true
sleep 2

# Secrets from /workspace/secrets.env (chmod 600, never committed): VLLM_API_KEY=..., CV_API_KEY=...
SECRETS="${SECRETS_FILE:-/workspace/secrets.env}"
if [[ ! -f "$SECRETS" ]]; then
  echo "error: $SECRETS not found (needs VLLM_API_KEY=... and CV_API_KEY=...)" >&2
  exit 1
fi
chmod 600 "$SECRETS"
set -a
# shellcheck disable=SC1090
source "$SECRETS"
set +a
if [[ -z "${VLLM_API_KEY:-}" || ${#VLLM_API_KEY} -lt 24 ]]; then
  echo "error: VLLM_API_KEY missing or shorter than 24 characters in $SECRETS" >&2
  exit 1
fi
echo "==> API key loaded from $SECRETS"

# Start supervisord daemon
echo "==> Starting supervisord..."
supervisord -c /workspace/forte/configs/supervisord.conf

echo "==> Waiting for vLLM & Nginx to become healthy on port 8000..."
READY=0
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 3
done

if [[ $READY -eq 1 ]]; then
  echo "==> SUCCESS: Production vLLM service is UP and healthy!"
  supervisorctl -c /workspace/forte/configs/supervisord.conf status
else
  echo "==> ERROR: Service did not start in time. Checking logs:"
  tail -n 40 /workspace/logs/vllm.log
  exit 1
fi
