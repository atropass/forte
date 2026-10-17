#!/usr/bin/env bash
# Switch active model in vLLM supervisor configuration
set -euo pipefail

TARGET="${1:-dev}"

if [[ "$TARGET" == "dev" ]]; then
  MODEL="Qwen/Qwen2-VL-2B-Instruct"
  MEM_UTIL="0.60"
  MAX_LEN="4096"
  MAX_SEQS="16"
  echo "==> Selecting Dev Model: $MODEL (Lightweight 2.2B VLM)"
elif [[ "$TARGET" == "prod" ]]; then
  MODEL="Qwen/Qwen3-VL-4B-Instruct"
  MEM_UTIL="0.85"
  MAX_LEN="8192"
  MAX_SEQS="32"
  echo "==> Selecting Prod Model: $MODEL (Production 4B VLM)"
else
  MODEL="$TARGET"
  MEM_UTIL="0.80"
  MAX_LEN="8192"
  MAX_SEQS="16"
  echo "==> Selecting Custom Model: $MODEL"
fi

CONF="/workspace/forte/configs/supervisord.conf"

# Update model in supervisord.conf
sed -i -E "s|--model [^ ]+|--model $MODEL|" "$CONF"
sed -i -E "s|--gpu-memory-utilization [^ ]+|--gpu-memory-utilization $MEM_UTIL|" "$CONF"
sed -i -E "s|--max-model-len [^ ]+|--max-model-len $MAX_LEN|" "$CONF"
sed -i -E "s|--max-num-seqs [^ ]+|--max-num-seqs $MAX_SEQS|" "$CONF"

echo "==> Updated supervisord configuration with model: $MODEL"
echo "==> Reloading vLLM via supervisorctl..."
supervisorctl -c "$CONF" update || true
supervisorctl -c "$CONF" restart vllm || true

echo "==> Model switched! Tail logs with: tail -f /workspace/logs/vllm.log"
