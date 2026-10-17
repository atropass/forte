"""Admin CLI: accounts, API keys, migration from the v1 SQLite store, purge."""
import hashlib
import sqlite3


def test_users_and_keys(svc_env, capsys):
    from service import admin, auth
    assert admin.main(["create-user", "jury"]) == 0
    out = capsys.readouterr().out
    pw = out.rsplit("password: ", 1)[1].strip()
    assert auth.check_password("jury", pw) == "jury"
    assert admin.main(["create-user", "jury", "--password", "x" * 20]) == 2  # exists
    assert admin.main(["create-user", "weak", "--password", "short"]) == 2
    assert admin.main(["list-users"]) == 0 and "jury" in capsys.readouterr().out
    assert admin.main(["create-api-key", "--name", "jury"]) == 0
    key = capsys.readouterr().out.strip().splitlines()[-1]
    assert auth.api_key_name(key) == "jury"
    admin.main(["list-api-keys"])
    assert key not in capsys.readouterr().out  # keys are shown once, never listed
    assert admin.main(["revoke-api-key", "jury"]) == 0
    assert auth.api_key_name(key) is None


def test_import_sqlite_users(svc_env, tmp_path, capsys):
    from service import admin, auth
    db = sqlite3.connect(tmp_path / "users.db")
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, salt BLOB, pw_hash BLOB, created REAL)")
    salt = b"s" * 16
    db.execute("INSERT INTO users (username, salt, pw_hash, created) VALUES (?, ?, ?, ?)",
               ("old_user", salt, hashlib.scrypt(b"old-password", salt=salt, n=2**14, r=8, p=1, dklen=32), 1.0))
    db.commit()
    assert admin.main(["import-sqlite-users", str(tmp_path / "users.db")]) == 0
    assert auth.check_password("old_user", "old-password") == "old_user"


def test_purge(svc_env, capsys):
    from service import admin, blob
    b = blob.get_blob()
    b.put_bytes(blob.job_key("abcd1234", "result.json"), b"{}")
    assert admin.main(["purge", "--older-than-days", "-1"]) == 0
    assert b.get_bytes(blob.job_key("abcd1234", "result.json")) is None
