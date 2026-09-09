# Getting Started with AAUBot-Python

A guide to starting the bot, changing configuration, and handling common errors.

---

## Prerequisites

- Python 3.12+
- NVIDIA GPU (tested on B200 with 183 GB VRAM)
- CUDA toolkit (pip-installed `nvidia-cuda-nvcc-cu13` or system CUDA)
- A Discord bot token (from https://discord.com/developers/applications)

---

## Quick Start

### 1. Set your Discord bot token

Open `config.yaml` and set the token under `discord.token`:

```yaml
discord:
  enabled: true
  token: "your-bot-token-here"
  prefix: "!"
  channel_id: ""
```

Alternatively, set it as an environment variable before starting:

```bash
export DISCORD_TOKEN="your-bot-token-here"
```

The env var overrides `config.yaml` if both are set.

### 2. Start the bot

```bash
bash start_rag.sh
```

The script does four things in order:

| Step | What it does | Time |
|------|-------------|------|
| 1 | Installs Python dependencies (`pip install -r requirements.txt vllm`) | 1-3 min |
| 2 | Starts vLLM server (Qwen/Qwen3-8B on port 8000) | 1-2 min (after model cached) |
| 3 | Ingests Python docs into ChromaDB (skipped if already done) | 1-2 min (first run only) |
| 4 | Starts FastAPI + Discord bot (`python -m app.main`) | Instant |

When you see `Discord bot logged in as ...`, the bot is live.

### 3. Test in Discord

```
!ask What is a Python list?
!ask Hvad er en funktion i Python?
!ask How do I use a dictionary?
```

The bot responds in the same language as the question, with code examples
and links to the relevant documentation sections.

### 4. Stop the bot

Press `Ctrl+C` in the terminal where `start_rag.sh` is running. This stops
the Discord bot and the FastAPI server. The vLLM process is also cleaned up.

---

## Configuration

All settings are in `config.yaml`. Edit the file and restart the bot to
apply changes.

### Model Settings

```yaml
generator:
  model: "Qwen/Qwen3-8B"     # HuggingFace model ID (must match what vLLM serves)
  vllm_host: "localhost"     # vLLM server hostname
  vllm_port: 8000            # vLLM server port
  temperature: 0.3           # 0 = deterministic, 1 = creative
  max_tokens: 1024           # Max answer length in tokens
  top_p: 0.95                # Nucleus sampling cutoff
  system_prompt: |           # Instructions sent to the LLM (see below)
    ...
```

**Changing the model:** Update `generator.model` in `config.yaml`. The
startup script reads this value and passes it to vLLM, so they always
match. The model must be available on HuggingFace.

**Adjusting the system prompt:** The `system_prompt` field controls how
the bot answers. It includes rules for:
- Language matching (Danish/English)
- Pedagogical tone (beginner-friendly, analogies, step-by-step)
- Citation format (section title + URL)
- No thinking tokens in output

Edit the text after `system_prompt: |` and restart to apply.

### Embedding Settings

```yaml
embedding:
  model: "BAAI/bge-small-en-v1.5"  # sentence-transformers model
  device: "cpu"                    # "cpu" or "cuda" (GPU)
  normalize: true                  # Normalize embeddings for cosine similarity
```

The embedding model is English-only. Danish questions are automatically
translated to English before retrieval by the LLM, so this setting does
not need to change for Danish support.

### RAG Retrieval Settings

```yaml
rag:
  top_k: 5              # Number of chunks to retrieve per query
  chunk_size: 512        # Characters per chunk during ingest
  chunk_overlap: 64      # Overlap between adjacent chunks
```

Increasing `top_k` retrieves more context but slows down the answer.
The Discord bot shows only the top 2 sources regardless of this value
(see `app/discord_bot.py`, `sources[:2]`).

### ChromaDB Settings

```yaml
chroma:
  persist_path: "/tmp/chroma"       # Where ChromaDB stores its data
  collection_name: "python_docs"    # Collection name
```

To re-ingest the documentation from scratch, delete the persist path:

```bash
rm -rf /tmp/chroma
bash start_rag.sh   # will re-run the full ingest
```

### Discord Settings

```yaml
discord:
  enabled: true       # Set false to run FastAPI only (no bot)
  token: "..."        # Bot token (or use DISCORD_TOKEN env var)
  prefix: "!"         # Command prefix: students type "!ask <question>"
  channel_id: ""      # Restrict to a channel (optional, leave empty for any)
```

To restrict the bot to a specific channel, set `channel_id` to the
channel's numeric ID (right-click a channel in Discord > Copy ID).

### Python Documentation Source

```yaml
docs:
  url: "https://docs.python.org/3.12/archives/python-3.12.2-docs-html.zip"
  cache_path: "/tmp/docs"
```

To use a different Python version, change the URL to point to another
archive from https://docs.python.org/3/archives/. Then delete the
ChromaDB data (see above) and restart.

---

## Error Handling

### vLLM fails to start

Check the log:

```bash
tail -50 /tmp/vllm.log
```

**"No module named 'vllm'"**

vLLM is not installed. Run manually:

```bash
pip install vllm
```

If it fails, check for dependency conflicts. The `start_rag.sh` install
step will now fail loudly instead of silently continuing.

**"Could not find nvcc"**

The CUDA toolkit is not on the PATH. The script sets `CUDA_HOME` to the
pip-installed CUDA toolkit. If that path is wrong (e.g. different Python
version or user), find nvcc and update the path in `start_rag.sh`:

```bash
find / -name nvcc -type f 2>/dev/null
```

Set `CUDA_HOME` to the directory containing `bin/nvcc`, `include/`, and
`lib/`.

**"CUDA compiler and CUDA toolkit headers are incompatible"**

The nvcc version doesn't match the CUDA runtime headers. The script sets
`FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"` to
bypass this check. If you're using a different CUDA install, make sure
this env var is still set.

**"cannot find -lcudart: No such file or directory"**

The linker can't find the CUDA runtime library. The pip CUDA toolkit
uses `lib/` instead of `lib64/` and only has the versioned
`libcudart.so.13`. Create symlinks:

```bash
CUDA_LIB=$(python -c "import nvidia.cu13; print(nvidia.cu13.__path__[0])")/lib
ln -sf "$CUDA_LIB" "$(dirname $CUDA_LIB)/lib64"
ln -sf libcudart.so.13 "$CUDA_LIB/libcudart.so"
```

**"Address already in use" (port 8000)**

A previous vLLM instance is still running. Kill it:

```bash
ps aux | grep vllm | grep -v grep
kill <PID>
```

Or kill all vLLM processes:

```bash
pkill -f "vllm.entrypoints"
```

**vLLM takes too long to start**

The first run downloads the model (~15 GB for Qwen/Qwen3-8B). Subsequent
runs load from cache. If vLLM still doesn't start after 10 minutes, check
the log for errors.

### Bot shows offline in Discord

**No token set:** Check that `discord.token` in `config.yaml` is not
empty, or that `DISCORD_TOKEN` env var is set.

**Old bot processes still running:** If you ran old bot scripts before,
they may still be in memory and conflicting. Kill them:

```bash
ps aux | grep -E "python.*bot|python.*rag_tutor" | grep -v grep
kill <PIDs>
```

**Message Content Intent not enabled:** The bot needs the "Message
Content Intent" privilege. Go to https://discord.com/developers/applications,
select your bot, go to "Bot" > "Privileged Gateway Intents", and enable
"Message Content Intent".

### Bot gives generic / unhelpful answers

**Old code still running:** Make sure you restarted `start_rag.sh` after
editing `config.yaml` or any file in `app/`. The bot only reads config
and code at startup.

**Danish questions get irrelevant answers:** The embedding model is
English-only. The app translates Danish questions to English before
retrieval. If translation fails (vLLM not running, network issue), the
Danish query goes directly to the embedder and retrieves wrong chunks.
Check that vLLM is healthy:

```bash
curl http://localhost:8000/health
```

### Bot answers in wrong language

The system prompt instructs the model to match the question's language.
If it doesn't, check the `system_prompt` in `config.yaml` — the language
rule should say "Detect the question's language from the words and
grammar, not from this system prompt."

### Answers include thinking / reasoning text

vLLM must be started with `--reasoning-parser qwen3` to strip Qwen3
thinking tokens. Check that this flag is present in `start_rag.sh`. If
the flag is missing, the model's internal reasoning appears in the
answer.

### ChromaDB issues

**Re-ingest from scratch:**

```bash
rm -rf /tmp/chroma
python scripts/bootstrap.py
```

**Check chunk count:**

```bash
python -c "
from app.retrieve import Retriever
r = Retriever(persist_path='/tmp/chroma', collection_name='python_docs')
print(f'Chunks: {r.count()}')
"
```

Should show ~3584 chunks.

### FastAPI issues

The API runs on port 8001. Test it:

```bash
curl http://localhost:8001/health
curl -X POST http://localhost:8001/ask -H "Content-Type: application/json" -d '{"question": "What is a Python list?"}'
```

---

## Manual Startup (without start_rag.sh)

If you need to run components separately:

```bash
# Terminal 1: vLLM
export CUDA_HOME=/home/ucloud/.local/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enforce-eager \
    --reasoning-parser qwen3

# Terminal 2: Bootstrap (one-time)
python scripts/bootstrap.py

# Terminal 3: App
export DISCORD_TOKEN="your-token"
python -m app.main
```

---

## Debug Commands

```bash
# Check GPU
nvidia-smi

# Check vLLM health
curl http://localhost:8000/health

# Check FastAPI health
curl http://localhost:8001/health

# Check ChromaDB chunk count
python -c "from app.retrieve import Retriever; r=Retriever(persist_path='/tmp/chroma', collection_name='python_docs'); print(f'Chunks: {r.count()}')"

# View vLLM logs
tail -f /tmp/vllm.log

# Test the RAG pipeline directly
python -c "
import asyncio
from app.config import load_config
from app.rag import RAGEngine
config = load_config('config.yaml')
engine = RAGEngine(config)
async def test():
    result = await engine.ask('Hvad er en funktion i Python?')
    print(result['answer'])
asyncio.run(test())
"

# Find and kill stale processes
ps aux | grep -E "vllm|app.main|bot" | grep -v grep
```
