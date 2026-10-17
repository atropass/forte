#!/bin/bash
# Fly.io single-machine supervisor: Redis (localhost only, AOF on the volume), then the workers and the API.
# If any of the three exits, the script exits and Fly restarts the machine.
set -euo pipefail
DATA="${DATA_DIR:-/data}"
mkdir -p "$DATA/redis"
chown -R app:app "$DATA"

setpriv --reuid=app --regid=app --init-groups \
  redis-server --bind 127.0.0.1 --port 6379 --dir "$DATA/redis" --appendonly yes --appendfsync everysec \
  --maxmemory-policy noeviction --save 300 10 --protected-mode yes &
for _ in $(seq 1 50); do redis-cli -h 127.0.0.1 ping >/dev/null 2>&1 && break; sleep 0.2; done

entrypoint admin init-storage
# one-off migration of the v1 accounts (SQLite on the volume); the file is renamed so it runs only once
if [ -f "$DATA/users.db" ]; then
  entrypoint admin import-sqlite-users "$DATA/users.db" && mv "$DATA/users.db" "$DATA/users.db.imported"
fi

entrypoint worker &
entrypoint api &
wait -n
echo "fly-start: a process exited, stopping the machine" >&2
exit 1
