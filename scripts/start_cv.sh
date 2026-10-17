#!/bin/bash
set -e

echo "==> [ForteVision] Initializing Computer Vision Microservice..."

# Stop system nginx so custom gateway nginx on port 8000 binds cleanly
service nginx stop 2>/dev/null || true
killall -9 nginx 2>/dev/null || true
sleep 1

# Clean any stale Hugging Face lock files from previous unclean shutdowns
find /root/.cache/huggingface -name '*.lock' -delete 2>/dev/null || true

# Launch supervisord daemon if not already running
if ! pgrep -f "supervisord -c /workspace/forte/configs/supervisord.cv.conf" > /dev/null; then
    CV_API_KEY="fv-prod-secret-forte-2026" /usr/bin/supervisord -c /workspace/forte/configs/supervisord.cv.conf
    echo "==> Supervisord started successfully."
else
    echo "==> Supervisord is already running."
    supervisorctl -c /workspace/forte/configs/supervisord.cv.conf restart all || true
fi
