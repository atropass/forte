# Analysis service image: API (`api`), workers (`worker`), or both + Redis in one container (`allinone`).
# The GPU models are NOT in this image: workers call the vLLM server and the CV service over HTTPS.
#   docker build -f docker/service.Dockerfile -t forte-vision .
#   docker run ... forte-vision api | worker | allinone | admin <args>
# Optional local CPU detector (no CV pod): --build-arg WITH_LOCAL_DETECTOR=1

FROM node:20.18-alpine AS ui
WORKDIR /ui
# corepack in node 20.18 fails pnpm's signature check ("Cannot find matching keyid"), so install pnpm via npm
RUN npm install -g pnpm@10.13.1
COPY forte-vision-ai-development/package.json forte-vision-ai-development/pnpm-lock.yaml forte-vision-ai-development/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY forte-vision-ai-development/ ./
ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm typecheck && pnpm build

FROM python:3.11-slim AS runtime
ARG WITH_LOCAL_DETECTOR=0
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core tini redis-server supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt
ENV HF_HOME=/app/hf
RUN if [ "$WITH_LOCAL_DETECTOR" = "1" ]; then \
      pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
      && pip install --no-cache-dir transformers==4.57.1 huggingface_hub safetensors \
      && python -c "from huggingface_hub import snapshot_download as d; d('PekingU/rtdetr_v2_r18vd')"; \
    fi

COPY pipeline/ pipeline/
COPY service/ service/
COPY legal/ legal/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint
COPY configs/supervisord.allinone.conf /etc/supervisor/allinone.conf
COPY --from=ui /ui/out /app/ui
RUN chmod 0755 /usr/local/bin/entrypoint && mkdir -p /data && chown -R app:app /data

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1 \
    APP_ENV=production LOG_FORMAT=json DATA_DIR=/data FRONTEND_DIR=/app/ui TMPDIR=/data/tmp \
    VLM_BACKEND=openai VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct PORT=8000 WEB_CONCURRENCY=2
# The entrypoint starts as root only to fix ownership of a freshly mounted volume, then drops to user `app`
# (uid 10001) for every process. Platforms that mount volumes writable for 10001 can run with --user 10001.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)" || exit 1
ENTRYPOINT ["tini", "--", "entrypoint"]
CMD ["api"]
