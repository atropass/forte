# Fly.io, the cheapest layout: ONE machine runs Redis + API (also serving the static UI) + workers
# (docker/fly-start.sh), sharing the `forte_data` volume. The GPU models stay on the team's pods
# (VLM_API_BASE / DET_API_BASE). docker-compose.yml is the multi-container equivalent.
#   fly deploy   (see fly.toml)

FROM node:20.18-alpine AS ui
WORKDIR /ui
# corepack in node 20.18 fails pnpm's signature check ("Cannot find matching keyid"), so install pnpm via npm
RUN npm install -g pnpm@10.13.1
COPY forte-vision-ai-development/package.json forte-vision-ai-development/pnpm-lock.yaml forte-vision-ai-development/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY forte-vision-ai-development/ ./
ARG NEXT_PUBLIC_MAX_UPLOAD_MB=100
ENV NEXT_TELEMETRY_DISABLED=1 NEXT_PUBLIC_MAX_UPLOAD_MB=$NEXT_PUBLIC_MAX_UPLOAD_MB
RUN pnpm typecheck && pnpm build

FROM python:3.11-slim-bookworm
# ffmpeg >= 5 is required (pipeline/frames.py reads stream_side_data); bookworm ships 5.1
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core tini redis-server \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY pipeline/ pipeline/
COPY service/ service/
COPY legal/ legal/
COPY --from=ui /ui/out /app/ui
COPY docker/entrypoint.sh /usr/local/bin/entrypoint
COPY docker/fly-start.sh /usr/local/bin/fly-start
RUN chmod 0755 /usr/local/bin/entrypoint /usr/local/bin/fly-start && mkdir -p /data && chown app:app /data

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1 \
    APP_ENV=production LOG_FORMAT=json DATA_DIR=/data TMPDIR=/tmp FRONTEND_DIR=/app/ui \
    REDIS_URL=redis://127.0.0.1:6379/0 STORAGE_BACKEND=fs \
    VLM_BACKEND=openai VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct PORT=8000
EXPOSE 8000
ENTRYPOINT ["tini", "--", "fly-start"]
