#!/bin/bash
set -e

echo "=== AAUBot-Python Starting ==="

# Step 1: Bootstrap — fetch docs, embed, ingest into ChromaDB (idempotent)
echo "[1/2] Running bootstrap (fetch docs, ingest)..."
python scripts/bootstrap.py

# Step 2: Wait for vLLM to be ready, then start API + Discord bot
VLLM_HOST="${VLLM_HOST:-vllm}"
VLLM_PORT="${VLLM_PORT:-8000}"
echo "[2/2] Waiting for vLLM at http://${VLLM_HOST}:${VLLM_PORT}/health..."
until python -c "
import urllib.request, sys
try:
    urllib.request.urlopen('http://${VLLM_HOST}:${VLLM_PORT}/health', timeout=5)
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null; do
    echo "  vLLM not ready, retrying in 5s..."
    sleep 5
done

echo "  vLLM is ready."

# Check test mode from config.yaml
TEST_MODE=$(python -c "import yaml; print(str(yaml.safe_load(open('config.yaml')).get('test', {}).get('enabled', False)).lower())")

if [ "$TEST_MODE" = "true" ]; then
    echo "  Test mode ON — starting Chainlit browser UI (no Discord)..."
    exec chainlit run app/chainlit_app.py --host 0.0.0.0 --port 8001
else
    echo "  Starting API + Discord bot..."
    exec python -m app.main
fi
