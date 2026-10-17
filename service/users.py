"""User accounts and login sessions in SQLite (DATA_DIR/users.db, on the persistent volume).

Open registration: username + password. Passwords are hashed with scrypt (stdlib). A session is a random
token kept in an HttpOnly cookie; only its SHA-256 is stored, so a leaked DB does not leak live sessions.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

SESSION_TTL_S = 30 * 24 * 3600
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD = 6
_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}


class AuthError(ValueError):
    pass


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class UserStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self.db:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    salt BLOB NOT NULL, pw_hash BLOB NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
                    created REAL NOT NULL, expires REAL NOT NULL);
            """)

    def register(self, username: str, password: str) -> str:
        username = username.strip()
        if not USERNAME_RE.match(username):
            raise AuthError("Логин: 3–32 символа, латиница, цифры, _ . -")
        if len(password) < MIN_PASSWORD:
            raise AuthError(f"Пароль: минимум {MIN_PASSWORD} символов")
        salt = secrets.token_bytes(16)
        try:
            with self._lock, self.db:
                self.db.execute("INSERT INTO users (username, salt, pw_hash, created) VALUES (?, ?, ?, ?)",
                                (username, salt, _hash_password(password, salt), time.time()))
        except sqlite3.IntegrityError:
            raise AuthError("Такой логин уже занят") from None
        return username

    def check_password(self, username: str, password: str) -> str:
        with self._lock:
            row = self.db.execute("SELECT username, salt, pw_hash FROM users WHERE username = ?",
                                  (username.strip(),)).fetchone()
        if row is None:
            _hash_password(password, b"0" * 16)  # same cost as a real check: no username probing by timing
            raise AuthError("Неверный логин или пароль")
        name, salt, pw_hash = row
        if not hmac.compare_digest(_hash_password(password, salt), pw_hash):
            raise AuthError("Неверный логин или пароль")
        return name

    def create_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock, self.db:
            self.db.execute("DELETE FROM sessions WHERE expires < ?", (now,))
            self.db.execute("INSERT INTO sessions (token_hash, user_id, created, expires) "
                            "SELECT ?, id, ?, ? FROM users WHERE username = ?",
                            (_token_hash(token), now, now + SESSION_TTL_S, username))
        return token

    def session_user(self, token: str) -> str | None:
        with self._lock:
            row = self.db.execute("SELECT u.username FROM sessions s JOIN users u ON u.id = s.user_id "
                                  "WHERE s.token_hash = ? AND s.expires > ?", (_token_hash(token), time.time())).fetchone()
        return row[0] if row else None

    def delete_session(self, token: str):
        with self._lock, self.db:
            self.db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
