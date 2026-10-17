"""Administration CLI (run where REDIS_URL / storage env is set, e.g. `fly ssh console` or `docker compose exec api`).

  python -m service.admin create-user <username> [--password ...]   (random password printed if omitted)
  python -m service.admin reset-password <username> [--password ...]
  python -m service.admin delete-user <username>
  python -m service.admin list-users
  python -m service.admin create-api-key --name jury                (key printed ONCE; only its hash is stored)
  python -m service.admin list-api-keys
  python -m service.admin revoke-api-key <name-or-id>
  python -m service.admin purge [--older-than-days N]               (results/logs in storage; default JOB_TTL_DAYS)
  python -m service.admin init-storage                              (create the S3 bucket if missing)
  python -m service.admin import-sqlite-users /data/users.db        (one-off migration from the v1 SQLite store)
"""
from __future__ import annotations

import argparse
import getpass
import secrets
import sys
import time

from . import auth, blob
from .config import get_settings


def _password(given: str | None, ask: bool) -> tuple[str, bool]:
    if given:
        return given, False
    if ask and sys.stdin.isatty():
        p = getpass.getpass("password (empty = generate): ")
        if p:
            return p, False
    return secrets.token_urlsafe(18), True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m service.admin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("create-user", "reset-password"):
        p = sub.add_parser(name)
        p.add_argument("username")
        p.add_argument("--password")
        p.add_argument("--ask", action="store_true", help="prompt for the password")
    sub.add_parser("delete-user").add_argument("username")
    sub.add_parser("list-users")
    sub.add_parser("create-api-key").add_argument("--name", required=True)
    sub.add_parser("list-api-keys")
    sub.add_parser("revoke-api-key").add_argument("name_or_id")
    p = sub.add_parser("purge")
    p.add_argument("--older-than-days", type=float)
    sub.add_parser("init-storage")
    sub.add_parser("import-sqlite-users").add_argument("path")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "create-user":
            pw, generated = _password(a.password, a.ask)
            auth.create_user(a.username, pw)
            print(f"user {a.username} created" + (f"; password: {pw}" if generated else ""))
        elif a.cmd == "reset-password":
            pw, generated = _password(a.password, a.ask)
            auth.set_password(a.username, pw)
            print(f"password of {a.username} changed; sessions revoked" + (f"; password: {pw}" if generated else ""))
        elif a.cmd == "delete-user":
            auth.delete_user(a.username)
            print(f"user {a.username} deleted")
        elif a.cmd == "list-users":
            print("\n".join(auth.list_users()) or "(no users)")
        elif a.cmd == "create-api-key":
            key = auth.create_api_key(a.name)
            print(f"API key for {a.name} (shown once, store it securely):\n{key}")
        elif a.cmd == "list-api-keys":
            for k in auth.list_api_keys():
                print(f"{k['id']}  {k['name']:<20} {k.get('prefix', '')}…  created "
                      f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(float(k['created'])))}")
            if get_settings().api_keys:
                print(f"(+ {len(get_settings().api_keys)} bootstrap key(s) from env API_KEYS/API_KEY)")
        elif a.cmd == "revoke-api-key":
            n = auth.revoke_api_key(a.name_or_id)
            print(f"revoked {n} key(s)")
            return 0 if n else 1
        elif a.cmd == "purge":
            days = a.older_than_days if a.older_than_days is not None else get_settings().job_ttl_s / 86400
            n = blob.get_blob().delete_older_than("jobs/", days * 86400)
            print(f"deleted {n} stored file(s) older than {days:g} days")
        elif a.cmd == "init-storage":
            b = blob.get_blob()
            if isinstance(b, blob.S3Blob):
                b.ensure_bucket()
            b.check()
            print("storage ok")
        elif a.cmd == "import-sqlite-users":
            import sqlite3
            n = skipped = 0
            for name, salt, pw_hash, created in sqlite3.connect(a.path).execute(
                    "SELECT username, salt, pw_hash, created FROM users"):
                if auth.import_user(name, bytes(salt), bytes(pw_hash), float(created)):
                    n += 1
                else:
                    skipped += 1
            print(f"imported {n} user(s), {skipped} already existed (passwords unchanged; sessions not migrated)")
    except auth.AuthError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
