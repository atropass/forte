#!/bin/sh
# Backend container roles:
#   api        FastAPI (uvicorn); WEB_CONCURRENCY processes (default 2). Internal only: the frontend nginx proxies it.
#   worker     RQ worker supervisor (WORKER_COUNT processes)
#   admin ...  python -m service.admin ...
set -e
if [ "$(id -u)" = "0" ]; then
  # root only to make a freshly mounted volume writable, then drop privileges for good
  mkdir -p "${DATA_DIR:-/data}"
  chown app:app "${DATA_DIR:-/data}"
  exec setpriv --reuid=app --regid=app --init-groups "$0" "$@"
fi

case "${1:-api}" in
  api)
    # proxy headers are trusted from FORWARDED_ALLOW_IPS; the api port must only be reachable by the proxy
    exec python -m uvicorn service.app:app --host 0.0.0.0 --port "${PORT:-8000}" \
      --workers "${WEB_CONCURRENCY:-2}" --timeout-keep-alive 75 --no-server-header \
      --proxy-headers --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-*}"
    ;;
  worker)
    exec python -m service.worker_main
    ;;
  admin)
    shift
    exec python -m service.admin "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
