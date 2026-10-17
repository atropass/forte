# Security and data handling

## Data
| Data | Where | Retention |
|---|---|---|
| Uploaded video | object storage `jobs/<id>/input.mov` (transient), worker temp dir | deleted right after analysis, also on failure or timeout; the reaper deletes it if a worker died |
| Result (verdict, violations, evidence frames) | object storage `jobs/<id>/result.json` | `JOB_TTL_DAYS` (default 30), then `service.admin purge` / bucket lifecycle rule |
| Job log | object storage `jobs/<id>/log.txt` | same as results; contains no video data, keys or passwords |
| Job record, users, sessions, API-key hashes | Redis | jobs: `JOB_TTL_DAYS`; sessions: `SESSION_TTL_H` |

* No face recognition, no identification, no audio. Roles come from position and behaviour only.
* No external AI services: the VLM and the detector run on team-controlled GPU servers.
* Test videos are never used for training, never archived, never logged.
* Recommended for S3: private bucket, `S3_SSE=AES256` (or KMS), lifecycle rule deleting `jobs/` after the retention.

## Authentication and authorization
* **API keys** (`X-API-Key`): issued with `service.admin create-api-key`, shown once, stored as SHA-256 in Redis,
  revocable at runtime. Optional bootstrap keys from env `API_KEYS` are compared in constant time.
  API-key callers see only jobs created with an API key.
* **Web UI**: accounts are created by an administrator (no self-registration). Passwords: scrypt (n=2^14, r=8, p=1,
  16-byte salt), minimum 12 characters for new accounts. Session: random 256-bit token in an `HttpOnly; Secure;
  SameSite=Strict` cookie; only its SHA-256 is stored; TTL `SESSION_TTL_H` (12 h); password reset revokes all sessions.
  A user sees only their own jobs; other users' jobs return 404 (indistinguishable from missing).
* **Brute force**: login is rate limited per IP (`LOGIN_IP_LIMIT` per `LOGIN_WINDOW_S`) and locked per username
  after `LOGIN_USER_FAIL_LIMIT` failures; failed checks cost the same time for unknown users.
* **CSRF**: cookie-authenticated state-changing requests must carry an `Origin`/`Referer` of `PUBLIC_ORIGIN`.

## HTTP hardening
* Headers on every response: `Content-Security-Policy` (`default-src 'self'`, `frame-ancestors 'none'`, no remote
  scripts), `Strict-Transport-Security`, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: same-origin`, `Permissions-Policy`, `Cache-Control: no-store` for API responses, `X-Request-ID`.
* Swagger/OpenAPI disabled unless `ENABLE_DOCS=1`; CORS off unless explicit origins are configured (never `*`).
* Upload limits enforced before reading the body (`Content-Length`) and while streaming; container signature check
  and `ffprobe` (file protocol only, timeout) with duration/resolution limits before a job is queued.
* Clients get generic error messages and error codes; details (tracebacks) stay in the server logs with the request id.
* Structured JSON logs; keys, passwords and file contents are never logged (covered by a test).

## Runtime
* Containers run as an unprivileged user (uid 10001); compose adds `read_only`, `cap_drop: ALL`,
  `no-new-privileges`. The image contains only runtime code (no tests, reports or datasets; see `.dockerignore`).
* Each analysis runs in its own process (RQ work-horse) with a hard time limit; a crash cannot affect the API or
  other analyses.
* Dependencies are pinned (`requirements-service.txt`, `pnpm-lock.yaml`); CI runs tests, the UI typecheck and a
  secret scan (gitleaks) on every push; deploy actions are pinned to commit SHAs.

## GPU servers
* nginx gateway in front of vLLM and the CV service: bearer tokens, 20 MB body limit, GET/POST only, no CORS,
  `/metrics` and API docs not exposed, rate limits.
* Tokens live in `/workspace/secrets.env` (chmod 600) on each server and in the deployment secrets; never in git,
  never on a command line. The CV service refuses to start without a key and never fetches remote URLs.

## Known limits / follow-ups
* The previous GPU gateway token was committed to git history: it must be rotated (see GPU_ACCESS.md); the repository
  should stay private.
* Rate limits on the GPU gateways key on the proxy address when the pod sits behind a provider proxy.
