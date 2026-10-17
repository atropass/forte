#!/usr/bin/env bash
# Boot script of the Secure Cloud standby pods (RunPod template start command):
#   nohup /workspace/forte/scripts/standby_boot.sh vllm|cv > /workspace/logs/boot.log 2>&1 & exec /start.sh
# Everything (venv, weights, code, /workspace/secrets.env) is already on the network volume; only the container
# packages are installed here, then the service is started. GPUs with < 23 GB get smaller vLLM settings.
set -euo pipefail
ROLE="${1:?usage: standby_boot.sh vllm|cv}"
echo "$(date -Is) standby boot: $ROLE"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends supervisor nginx curl procps >/dev/null
mkdir -p /workspace/logs
case "$ROLE" in
  vllm)
    CONF=/workspace/forte/configs/supervisord.conf
    MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "${MEM_MB:-0}" -lt 23000 ]; then
      echo "GPU memory ${MEM_MB} MB: --max-num-seqs 16, --gpu-memory-utilization 0.90"
      sed -i -E "s|--gpu-memory-utilization [^ ]+|--gpu-memory-utilization 0.90|; s|--max-num-seqs [^ ]+|--max-num-seqs 16|" "$CONF"
    else
      sed -i -E "s|--gpu-memory-utilization [^ ]+|--gpu-memory-utilization 0.85|; s|--max-num-seqs [^ ]+|--max-num-seqs 32|" "$CONF"
    fi
    exec bash /workspace/forte/scripts/setup_gpu_prod.sh
    ;;
  cv)
    exec bash /workspace/forte/scripts/start_cv.sh
    ;;
  *)
    echo "unknown role $ROLE" >&2; exit 2
    ;;
esac
