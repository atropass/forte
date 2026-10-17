# Backend image: API (`api`), workers (`worker`) and the admin CLI (`admin`). No UI, no models: workers call the
# vLLM server and the CV service over HTTPS (VLM_API_BASE / DET_API_BASE).
#   docker build -f docker/backend.Dockerfile -t forte-backend .
# Optional local CPU detector (runs without the CV service): --build-arg WITH_LOCAL_DETECTOR=1
FROM python:3.11-slim-bookworm

ARG WITH_LOCAL_DETECTOR=0
# ffmpeg >= 5 is required (pipeline/frames.py reads stream_side_data); bookworm ships 5.1
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && ffmpeg -loglevel error -f lavfi -i testsrc=size=64x64:rate=5:duration=1 -c:v mpeg4 /tmp/check.mp4 \
    && ffprobe -v error -select_streams v:0 -show_entries stream=width,height:stream_side_data=rotation:format=duration \
       -of json /tmp/check.mp4 > /dev/null \
    && rm /tmp/check.mp4

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
RUN chmod 0755 /usr/local/bin/entrypoint && mkdir -p /data && chown app:app /data

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1 \
    APP_ENV=production LOG_FORMAT=json DATA_DIR=/data TMPDIR=/tmp FRONTEND_DIR=/nonexistent \
    VLM_BACKEND=openai VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct PORT=8000 WEB_CONCURRENCY=2
# Starts as root only if a freshly mounted volume needs its owner fixed, then drops to `app` (uid 10001) for good.
# Compose / Kubernetes can run it directly as 10001.
EXPOSE 8000
ENTRYPOINT ["tini", "--", "entrypoint"]
CMD ["api"]
