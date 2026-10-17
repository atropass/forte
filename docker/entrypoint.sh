#!/bin/sh
# Container roles:
#   api        FastAPI (uvicorn); WEB_CONCURRENCY uvicorn processes (default 2)
#   worker     RQ worker supervisor (WORKER_COUNT processes)
#   allinone   Redis + api + workers under supervisord, state on /data (single host, e.g. one Fly machine)
#   admin ...  python -m service.admin ...
set -e
if [ "$(id -u)" = "0" ]; then
  # root only to make a freshly mounted volume writable, then drop privileges for good
  mkdir -p "${DATA_DIR:-/data}" "${TMPDIR:-/tmp}"
  chown app:app "${DATA_DIR:-/data}" "${TMPDIR:-/tmp}"
  [ -d "${DATA_DIR:-/data}/redis" ] && chown -R app:app "${DATA_DIR:-/data}/redis"
  [ -d "${DATA_DIR:-/data}/blobs" ] && chown app:app "${DATA_DIR:-/data}/blobs"
  exec setpriv --reuid=app --regid=app --init-groups "$0" "$@"
fi
mkdir -p "${DATA_DIR:-/data}" "${TMPDIR:-/tmp}" 2>/dev/null || true

case "${1:-api}" in
  api)
    exec python -m uvicorn service.app:app --host 0.0.0.0 --port "${PORT:-8000}" \
      --workers "${WEB_CONCURRENCY:-2}" --timeout-keep-alive 75 --proxy-headers --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-*}" \
      --no-server-header
    ;;
  worker)
    exec python -m service.worker_main
    ;;
  allinone)
    mkdir -p "${DATA_DIR:-/data}/redis"
    export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
    exec supervisord -n -c /etc/supervisor/allinone.conf
    ;;
  admin)
    shift
    exec python -m service.admin "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
