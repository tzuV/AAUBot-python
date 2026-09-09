# AAUBot-Python Architecture

A guide for future agents to understand and navigate the codebase.

---

## Overview

AAUBot-Python is a RAG (Retrieval-Augmented Generation) chatbot that answers
Python programming questions using official Python documentation. It runs
as a Discord bot backed by a vLLM-served LLM (Qwen/Qwen3-8B) and a ChromaDB
vector store.

There are two deployment paths:
- **Non-Docker** (`start_rag.sh`) — runs everything on a single VM with a GPU
- **Docker** (`docker-compose.yml`) — runs vLLM and the app in separate containers

---

## File Map

```
AAUBot-python/
├── start_rag.sh              # Non-Docker entry point (vLLM + bootstrap + app)
├── start.sh                  # Docker container entry point (bootstrap + app)
├── config.yaml               # Central configuration (model, RAG, Discord, etc.)
├── requirements.txt          # Python dependencies (app container only)
├── Dockerfile                # App container image
├── docker-compose.yml        # Orchestrates vllm + app containers
├── .env.example              # Template for environment variables
├── memory.md                 # Project memory and debugging history
├── get_started.md            # User-facing setup and troubleshooting guide
├── architecture.md           # This file
│
├── app/                      # Core application package
│   ├── __init__.py            #   Package marker
│   ├── config.py              #   YAML config loader + env var overrides
│   ├── main.py               #   FastAPI server + Discord bot entry point
│   ├── rag.py                #   RAG orchestrator (translate, embed, retrieve, generate)
│   ├── discord_bot.py        #   Discord bot with !ask command
│   ├── generate.py           #   vLLM HTTP client (generate, translate, health check)
│   ├── embeddings.py         #   sentence-transformers wrapper
│   ├── retrieve.py           #   ChromaDB vector store wrapper
│   ├── ingest.py             #   Document download, parse, chunk, and ingest
│   └── chainlit_app.py       #   Chainlit web UI (test mode only)
│
└── scripts/
    ├── bootstrap.py           #   Standalone ingest runner (idempotent)
    └── install.sh             #   Host-level NVIDIA/Docker setup script
```

---

## Module Dependency Graph

```
                     config.yaml
                          │
                    app/config.py
                 (load_config + env overrides)
                          │
         ┌────────────────┼────────────────────┐
         │                │                    │
   app/main.py    app/chainlit_app.py   scripts/bootstrap.py
         │                │                    │
         ├── app/rag.py   │              ┌─────┼─────┐
         │         │      │              │     │     │
         │    ┌────┼──┐   │         app/   app/   app/
         │    │    │  │   │      embeddings retrieve ingest
         │  embed  retr gen
         │  dings  ieve erator
         │         │
         └── app/discord_bot.py
```

**Key rule:** `app/rag.py` is the central hub. It instantiates and coordinates
`Embedder`, `Retriever`, and `Generator`. Everything else either calls
`RAGEngine.ask()` or feeds data into it.

---

## Module Reference

### app/config.py

**Purpose:** Loads `config.yaml` and applies environment variable overrides.

```python
def load_config(path: str = "config.yaml") -> dict
```

**Env var overrides:**
| Variable | Overrides config key |
|---|---|
| `VLLM_MODEL` | `generator.model` |
| `VLLM_HOST` | `generator.vllm_host` |
| `VLLM_PORT` | `generator.vllm_port` |
| `DISCORD_TOKEN` | `discord.token` + sets `discord.enabled = True` |

**Imported by:** `app/main.py`, `app/chainlit_app.py`, `scripts/bootstrap.py`

---

### app/rag.py

**Purpose:** The RAG orchestrator. Wires together embedder, retriever, and
generator into a single `ask()` pipeline.

```python
class RAGEngine:
    def __init__(self, config: dict)
    async def ask(self, question: str) -> dict
```

**`ask()` pipeline (4 steps):**

1. **Translate** — `generator.translate_to_english(question)` sends the
   question to vLLM with a translation prompt (temperature 0). Returns English
   text. Needed because the embedding model is English-only but the
   documentation is in English.

2. **Embed** — `embedder.embed_query(english_query)` converts the translated
   query to a vector using BAAI/bge-small-en-v1.5.

3. **Retrieve** — `retriever.query(embedding, top_k=5)` searches ChromaDB for
   the top-5 matching documentation chunks. Each chunk's context string
   includes its title and URL so the LLM can cite specific sections.

4. **Generate** — `generator.generate(question, context)` sends the original
   question (not the translated one) plus the retrieved context to vLLM. The
   system prompt controls language matching, pedagogy, and citation format.

**Returns:**
```python
{
    "answer": str,
    "sources": [{"title": str, "url": str}, ...]
}
```

**Config keys used:** `generator.*`, `embedding.*`, `rag.top_k`, `chroma.*`

**Imported by:** `app/main.py`, `app/chainlit_app.py`

---

### app/generate.py

**Purpose:** Async HTTP client for vLLM's OpenAI-compatible API. Three
operations: answer generation, question translation, and health check.

```python
class Generator:
    def __init__(self, model, vllm_host, vllm_port, temperature, max_tokens, top_p, system_prompt)
    async def generate(self, question: str, context: str) -> str
    async def translate_to_english(self, question: str) -> str
    async def is_ready(self) -> bool
```

- `generate()` — Replaces `{context}` in `system_prompt` with the retrieved
  context, POSTs to `/v1/chat/completions`, returns the answer text.
- `translate_to_english()` — POSTs a translation prompt (temp=0, max_tokens=256)
  to `/v1/chat/completions`. Returns English text. Used before retrieval for
  non-English questions.
- `is_ready()` — GETs `/health`. Used by FastAPI `/health` endpoint.

**HTTP client:** `httpx.AsyncClient` with 120-second timeout.

**Imported by:** `app/rag.py`

---

### app/embeddings.py

**Purpose:** Wraps sentence-transformers for encoding documents and queries.

```python
class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", normalize: bool = True)
    def embed(self, texts: list[str]) -> list[list[float]]
    def embed_query(self, text: str) -> list[float]
```

- `embed()` — Batch encode (used by `ingest.py` during document ingestion).
- `embed_query()` — Single-query encode (used by `rag.py` at query time).

**Model:** `BAAI/bge-small-en-v1.5` (English-only, ~130 MB).

**Imported by:** `app/rag.py`, `scripts/bootstrap.py`

---

### app/retrieve.py

**Purpose:** ChromaDB persistent vector store wrapper with cosine similarity.

```python
class Retriever:
    def __init__(self, persist_path: str, collection_name: str)
    def add(self, ids, embeddings, documents, metadatas)
    def query(self, query_embedding: list[float], top_k: int = 5) -> dict
    def count(self) -> int
```

- `add()` — Batch-adds documents in chunks of 500 (used by `ingest.py`).
- `query()` — Returns `{"documents", "metadatas", "distances"}`.
- `count()` — Returns total stored chunks (used to check if ingest is needed).

**Storage:** Persistent on disk at `/tmp/chroma`. Collection uses HNSW index
with cosine distance.

**Imported by:** `app/rag.py`, `scripts/bootstrap.py`

---

### app/ingest.py

**Purpose:** Document ingestion pipeline — download, parse, chunk, embed,
and store in ChromaDB.

```python
def download_docs(url: str, cache_path: str) -> str
def parse_html_files(extract_path: str) -> list[dict]
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]
def ingest(docs_url, cache_path, embedder, retriever, chunk_size, chunk_overlap) -> int
```

- `download_docs()` — Downloads and extracts the Python docs ZIP. Skips if
  already extracted.
- `parse_html_files()` — Walks all `.html` files, strips nav/script/style,
  extracts text content, builds source URLs relative to `docs.python.org/3/`.
  Returns list of `{"text", "title", "url"}`.
- `chunk_text()` — Splits text into ~512-char chunks at paragraph boundaries
  with 64-char overlap.
- `ingest()` — Orchestrates the full pipeline. Skips if `retriever.count() > 0`
  (idempotent). Embeds in batches of 256, stores in batches of 500.

**Imported by:** `scripts/bootstrap.py`

---

### app/main.py

**Purpose:** FastAPI server + Discord bot entry point for production mode.

```python
# Pydantic models
class AskRequest(BaseModel):   # question: str
class Source(BaseModel):        # title: str, url: str
class AskResponse(BaseModel):   # answer: str, sources: list[Source]

# Endpoints
GET  /          → service info
GET  /health    → checks vLLM readiness
POST /ask       → runs RAGEngine.ask(), returns AskResponse
```

**Startup logic (`main()`):**
1. Instantiate `RAGEngine(config)`
2. Configure uvicorn server on `api.host:api.port`
3. If `discord.enabled` and `discord.token`:
   - Run FastAPI + Discord bot concurrently via `asyncio.gather`
4. Else:
   - Run FastAPI only (prints "Discord bot disabled" message)

**Imported by:** Invoked by `start.sh` and `start_rag.sh` as `python -m app.main`

---

### app/discord_bot.py

**Purpose:** Discord bot that listens for `!ask <question>` commands and
forwards them to the RAG engine.

```python
async def run_discord_bot(rag_engine, discord_cfg: dict, use_message_content=True)
```

- Creates `commands.Bot` with `message_content` intent
- Registers `!ask` command that calls `rag_engine.ask(question)`
- Formats response as `**Answer:**\n{answer}\n**Sources:**\n[top 2 sources]`
- Chunks response to Discord's 2000-char message limit

**Config keys used:** `discord.prefix`, `discord.channel_id`, `discord.token`

**Imported by:** `app/main.py`

---

### app/chainlit_app.py

**Purpose:** Chainlit browser UI for testing the RAG bot without Discord.
Only used when `test.enabled: true` in `config.yaml`.

```python
async def on_chat_start()       # sends welcome message
async def on_message(message)   # calls rag_engine.ask(), shows answer + sources
```

**Imported by:** Invoked by `start.sh` as `chainlit run app/chainlit_app.py`

---

### scripts/bootstrap.py

**Purpose:** Standalone script that runs the document ingestion pipeline.
Idempotent — skips if ChromaDB already has data.

```python
def main()  # loads config, creates Embedder + Retriever, calls ingest()
```

**Imported by:** Called by `start.sh` and `start_rag.sh`

---

## Configuration

All configuration is in `config.yaml`. The structure:

```yaml
test:
  enabled: false              # true = Chainlit test UI, false = FastAPI + Discord

generator:                    # LLM settings
  model: "Qwen/Qwen3-8B"
  vllm_host: "localhost"
  vllm_port: 8000
  temperature: 0.3
  max_tokens: 1024
  top_p: 0.95
  system_prompt: |            # Multi-line prompt with {context} placeholder
    ...

embedding:                   # sentence-transformers settings
  model: "BAAI/bge-small-en-v1.5"
  device: "cpu"
  normalize: true

rag:                         # Retrieval settings
  top_k: 5
  chunk_size: 512
  chunk_overlap: 64

chroma:                      # Vector store settings
  persist_path: "/tmp/chroma"
  collection_name: "python_docs"

docs:                        # Python documentation source
  url: "https://docs.python.org/3.12/archives/python-3.12.2-docs-html.zip"
  cache_path: "/tmp/docs"

discord:                     # Bot settings
  enabled: true
  token: "..."
  prefix: "!"
  channel_id: ""

vllm:                        # vLLM server settings (Docker path)
  gpu_memory_utilization: 0.90
  max_model_len: 8192

api:                         # FastAPI server settings
  host: "0.0.0.0"
  port: 8001
```

**Environment variable overrides** (handled in `app/config.py`):

| Variable | Effect |
|---|---|
| `VLLM_MODEL` | Overrides `generator.model` |
| `VLLM_HOST` | Overrides `generator.vllm_host` |
| `VLLM_PORT` | Overrides `generator.vllm_port` |
| `DISCORD_TOKEN` | Overrides `discord.token`, sets `discord.enabled = true` |

---

## Startup Sequences

### Non-Docker (`start_rag.sh`)

```
start_rag.sh
  │
  ├── [1/4] pip install -r requirements.txt vllm
  │
  ├── [2/4] Start vLLM in background
  │         --model from config.yaml
  │         --enforce-eager          (skip torch AOT compilation)
  │         --reasoning-parser qwen3 (strip Qwen3 thinking tokens)
  │         Poll http://localhost:8000/health until ready
  │
  ├── [3/4] python scripts/bootstrap.py
  │         (skips if ChromaDB already has chunks)
  │
  └── [4/4] python -m app.main (foreground)
            ├── FastAPI on port 8001
            └── Discord bot (if token present)
            └── Ctrl+C stops everything, vLLM killed
```

**Environment variables set by the script:**
```bash
CUDA_HOME      # pip-installed CUDA 13 toolkit (for FlashInfer JIT on Blackwell)
FLASHINFER_EXTRA_CUDAFLAGS  # -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
```

### Docker (`docker-compose.yml` + `start.sh`)

```
docker compose up --build
  │
  ├── vllm container starts
  │   Image: vllm/vllm-openai:latest
  │   GPU passthrough, port 8000, healthcheck (300s start period)
  │
  └── app container starts (depends_on: vllm)
        └── start.sh
              ├── [1/2] python scripts/bootstrap.py
              ├── [2/2] Wait for http://vllm:8000/health
              └── if test.enabled:
                    exec chainlit run app/chainlit_app.py
                  else:
                    exec python -m app.main
```

---

## RAG Query Flow

This is the most important flow to understand. It happens inside
`RAGEngine.ask()` in `app/rag.py`:

```
User asks: "!ask hvad er en funktion i python?"
  │
  ▼
1. TRANSLATE (app/generate.py → Generator.translate_to_english)
   │  Sends question to vLLM: "Translate to English. Output ONLY the translation."
   │  temperature=0, max_tokens=256
   │  Returns: "what is a function in python?"
   │
   ▼
2. EMBED (app/embeddings.py → Embedder.embed_query)
   │  Encodes English query using BAAI/bge-small-en-v1.5
   │  Returns: list[float] (vector)
   │
   ▼
3. RETRIEVE (app/retrieve.py → Retriever.query)
   │  Searches ChromaDB for top-5 cosine-similar chunks
   │  Returns: documents + metadatas (title, url) + distances
   │
   ▼
4. BUILD CONTEXT (app/rag.py)
   │  Formats chunks as:
   │  "[1] 4. More Control Flow Tools\nURL: https://...\n<chunk text>"
   │  "[2] 6. Modules\nURL: https://...\n<chunk text>"
   │  ...
   │
   ▼
5. GENERATE (app/generate.py → Generator.generate)
   │  Sends to vLLM:
   │  - system: system_prompt with {context} replaced by the context string
   │  - user: original question (NOT the translated one)
   │  temperature=0.3, max_tokens=1024
   │  Returns: answer text
   │
   ▼
6. RETURN (app/rag.py)
   {"answer": "...", "sources": [{"title": "...", "url": "..."}, ...]}
   │
   ▼
7. DISCORD (app/discord_bot.py)
   Formats as:
   **Answer:**
   <answer text>

   **Sources:**
   - [title](url)     (top 2 only)
   - [title](url)
   Chunks to 2000-char Discord message limit.
```

**Why translate before retrieval?** The embedding model (BAAI/bge-small-en-v1.5)
is English-only. Danish queries produce poor embeddings that don't match the
English documentation chunks. The LLM translates the question to English for
retrieval, but the original question is sent to the generator so the answer
comes back in the right language.

---

## External Dependencies

| Package | Used by | Purpose |
|---|---|---|
| `discord.py` | `app/discord_bot.py` | Discord bot framework |
| `fastapi` + `uvicorn` | `app/main.py` | REST API server |
| `httpx` | `app/generate.py` | Async HTTP client for vLLM API |
| `sentence-transformers` | `app/embeddings.py` | Embedding model |
| `chromadb` | `app/retrieve.py` | Vector database |
| `beautifulsoup4` | `app/ingest.py` | HTML parsing of Python docs |
| `pyyaml` | `app/config.py` | YAML config loading |
| `pydantic` | `app/main.py` | Request/response models |
| `chainlit` | `app/chainlit_app.py` | Test web UI |
| `vllm` | `start_rag.sh` (non-Docker only) | LLM inference server |

---

## Key Design Decisions

1. **Translation before retrieval** — The embedding model is English-only, so
   Danish questions are translated to English by the LLM before embedding. The
   original question is still used for generation so the answer matches the
   question's language.

2. **Thinking token suppression** — Qwen3-8B emits reasoning tokens by default.
   vLLM is started with `--reasoning-parser qwen3` to separate these into a
   `reasoning_content` field that the app ignores. Only `content` is returned.

3. **`--enforce-eager` flag** — Skips torch AOT compilation, which requires
   nvcc at runtime. Needed because the pip-installed CUDA toolkit has version
   mismatches that break the compilation.

4. **Sources in context** — Each retrieved chunk's context string includes
   its URL (`[1] Title\nURL: https://...\n<text>`), so the LLM can reference
   specific sections instead of giving generic "check the docs" answers.

5. **Sources added by bot, not LLM** — The system prompt tells the LLM not to
   include citations. The Discord bot appends its own "Sources:" section with
   the top 2 retrieved chunks. This avoids duplicate citation sections.

6. **Idempotent bootstrap** — `scripts/bootstrap.py` checks
   `retriever.count() > 0` and skips ingestion if ChromaDB already has data.

7. **Config-driven model selection** — `start_rag.sh` reads the model ID from
   `config.yaml` and passes it to vLLM, ensuring the served model always
   matches what the app requests.

---

## Ports

| Service | Port | Purpose |
|---|---|---|
| vLLM | 8000 | OpenAI-compatible LLM API (`/health`, `/v1/chat/completions`) |
| FastAPI | 8001 | REST API (`/`, `/health`, `/ask`) |
| Chainlit | 8001 | Web UI (test mode, same port as FastAPI — mutually exclusive) |
