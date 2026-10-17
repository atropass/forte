# Fly.io (CPU): FastAPI + RT-DETRv2 detector + static UI. The VLM is NOT in this image:
# VLM_BACKEND=openai sends requests to the team's vLLM on the RunPod GPU pod (VLM_API_BASE / VLM_API_KEY secrets).
# Build context = clean `git archive` of the forte repo (see scripts/deploy_fly.sh), so .env / .venv / videos never get in.

FROM node:20.18-alpine AS ui
WORKDIR /ui
# corepack in node 20.18 fails pnpm's signature check ("Cannot find matching keyid"), so install pnpm via npm
RUN npm install -g pnpm@10.13.1
COPY forte-vision-ai-development/package.json forte-vision-ai-development/pnpm-lock.yaml forte-vision-ai-development/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile || pnpm install
COPY forte-vision-ai-development/ ./
ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt transformers scikit-image scikit-learn joblib huggingface_hub

# detector weights baked into the image (the service runs with HF_HUB_OFFLINE=1)
ENV HF_HOME=/app/hf
RUN python -c "from huggingface_hub import snapshot_download as d; d('PekingU/rtdetr_v2_r18vd')"

COPY pipeline/ pipeline/
COPY service/ service/
COPY legal/ legal/
COPY scripts/ scripts/
COPY data/labels.csv data/labels.csv
COPY --from=ui /ui/out /app/ui

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1 \
    VLM_BACKEND=openai VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct DET_MODEL=PekingU/rtdetr_v2_r18vd DEVICE=cpu \
    DATA_DIR=/data FRONTEND_DIR=/app/ui
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "75"]
