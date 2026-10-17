# GPU backend: FastAPI + vLLM (Qwen3-VL) + RT-DETRv2. Weights are mounted from ./weights (see
# scripts/download_weights.sh); the container runs with HF_HUB_OFFLINE=1, so no internet is needed.
FROM vllm/vllm-openai:v0.11.0

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt
COPY pipeline/ pipeline/
COPY service/ service/
COPY legal/ legal/
COPY scripts/ scripts/
COPY data/labels.csv data/labels.csv

ENV HF_HOME=/weights HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    VLM_BACKEND=vllm VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct DET_MODEL=PekingU/rtdetr_v2_r18vd \
    DATA_DIR=/data THRESHOLDS_FILE=/data/thresholds.yaml FRONTEND_DIR=/app/ui PYTHONUNBUFFERED=1
EXPOSE 8000
ENTRYPOINT []
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "75"]
