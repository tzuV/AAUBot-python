#!/bin/bash
# AAUBot RAG Startup Script - Non-Docker Edition
# Runs: vLLM (model from config.yaml) + ChromaDB + Discord Bot
# Usage: ./start_rag.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="/work/Python-RAG/AAUBot-python"
cd "$PROJECT_DIR"

# CUDA toolkit (pip-installed via nvidia-cuda-nvcc-cu13)
# Needed by FlashInfer JIT compilation on Blackwell (sm100) GPUs
export CUDA_HOME="/home/ucloud/.local/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
# Skip CCCL version check (pip nvcc 13.3 vs runtime headers 13.0 mismatch)
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

echo -e "${BLUE}=========================================="
# Read model ID from config.yaml so vLLM and the app always agree
MODEL=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['generator']['model'])")
echo -e "  AAUBot RAG Startup (Non-Docker)"
echo -e "  Model: ${MODEL}"
echo -e "  GPU: Using available NVIDIA GPU"
echo -e "==========================================${NC}"
echo

# ============================================
# Step 1: Install Dependencies
# ============================================
echo -e "${YELLOW}Step 1/4: Installing Python dependencies...${NC}"
if ! pip install -r requirements.txt vllm; then
    echo -e "${RED}❌ Dependency installation failed. See errors above.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo

# ============================================
# Step 2: Start vLLM Server
# ============================================
echo -e "${YELLOW}Step 2/4: Starting vLLM server...${NC}"
echo -e "${YELLOW}This will take a few minutes (model download + loading)${NC}"
echo

# Start vLLM in background
# --enforce-eager: skip torch AOT compilation (avoids nvcc dependency at that stage)
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enforce-eager \
    --reasoning-parser qwen3 \
    > /tmp/vllm.log 2>&1 &

VLLM_PID=$!
echo -e "vLLM process started (PID: $VLLM_PID)"
echo -e "Model is downloading and loading... (check /tmp/vllm.log)"

# Wait for vLLM to be ready
echo -e "${YELLOW}Waiting for vLLM to start...${NC}"
MAX_RETRIES=60
RETRY_DELAY=10
retry_count=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ vLLM server is ready at http://localhost:8000${NC}"
        break
    fi
    retry_count=$((retry_count + 1))
    echo -e "  Retry $retry_count/$MAX_RETRIES... (waiting ${RETRY_DELAY}s)"
    sleep $RETRY_DELAY
done

if [ $retry_count -ge $MAX_RETRIES ]; then
    echo -e "${RED}❌ vLLM failed to start. Check /tmp/vllm.log${NC}"
    echo -e "Common issues:"
    echo -e "  - CUDA_HOME not set correctly (check nvcc is on PATH)"
    echo -e "  - FlashInfer JIT compilation failed (check /tmp/vllm.log)"
    echo -e "  - Not enough GPU memory"
    echo -e "  - Out of disk space for model"
    exit 1
fi
echo

# ============================================
# Step 3: Ingest Python Documentation
# ============================================
echo -e "${YELLOW}Step 3/4: Ingesting Python documentation into ChromaDB...${NC}"
echo -e "${YELLOW}This will take 1-2 minutes${NC}"
echo

python scripts/bootstrap.py

echo -e "${GREEN}✅ Python documentation ingested into ChromaDB${NC}"
echo

# ============================================
# Step 4: Start Discord Bot with REAL RAG
# ============================================
echo -e "${YELLOW}Step 4/4: Starting Discord bot with REAL RAG...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo

echo -e "${BLUE}=========================================="
echo -e "  AAUBot RAG is now running!"
echo -e "  vLLM: http://localhost:8000"
echo -e "  Discord: PyTutor#6616"
echo -e "  FastAPI: http://localhost:8001"
echo -e "==========================================${NC}"
echo -e "\nTest in Discord with:"
echo -e "  !ask What is a Python list?"
echo -e "  !ask Hvad er en Python funktion?"
echo -e "\nPress Ctrl+C to stop everything\n"

# Start the main app (FastAPI + Discord bot)
# This will run in foreground, so Ctrl+C will stop it
python -m app.main

# If we get here, cleanup vLLM
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
