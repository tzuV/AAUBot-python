# AAUBot-Python RAG Project - Memory & Progress

> **Last Updated:** 2026-09-09
> **Status:** Working — RAG pipeline verified end-to-end, Discord bot live
> **Maintainer:** User + Mistral Vibe
> **GPU:** NVIDIA B200 (183GB VRAM)

---

## Project Purpose

**AAUBot-Python** is a RAG (Retrieval-Augmented Generation) chatbot for Danish
master's students in an IT-management course at Aalborg University (AAU). It
provides Python programming tutoring by:

1. Retrieving relevant chunks from official Python documentation (ChromaDB)
2. Generating beginner-friendly answers using Qwen/Qwen3-8B via vLLM
3. Delivering responses via Discord (`!ask <question>`)

### Key Objectives
- Answer Python questions using official documentation (not hallucinated)
- Support both English and Danish (auto-detect question language, translate for retrieval)
- Pedagogical explanations for students with no prior technical experience
- Run without Docker on a VM with NVIDIA B200 GPU

### Target Audience
- AAU students learning Python as part of IT-management curriculum
- Danish-speaking students who prefer native-language support
- Beginners who need clear, step-by-step explanations with analogies

---

## Current Status

### Completed

| Component | Status | Details |
|-----------|--------|---------|
| Discord Bot | Working | `app/discord_bot.py` with `!ask` command, top 2 sources shown |
| RAG Pipeline | Working | `app/rag.py` — embed, retrieve, translate, generate |
| vLLM Integration | Working | Qwen/Qwen3-8B on B200, `--enforce-eager --reasoning-parser qwen3` |
| ChromaDB | Working | 3584 chunks from Python 3.12.2 docs ingested |
| Embedding Model | Working | BAAI/bge-small-en-v1.5 on CPU |
| Danish Support | Working | LLM translates Danish questions to English before retrieval |
| Citations | Working | Answers include specific section titles + URLs |
| Pedagogical Prompt | Working | System prompt instructs no-jargon, analogies, step-by-step |
| Thinking Token Suppression | Working | `--reasoning-parser qwen3` separates reasoning from content |

### Not Started

| Task | Priority | Notes |
|------|----------|-------|
| User Analytics | Low | Track popular questions |
| Rate Limiting | Low | Prevent abuse |
| Web Interface | Low | Chainlit UI available in `app/chainlit_app.py` |

---

## Technical Architecture

### System Components

```
AAUBot-Python/
├── start_rag.sh          # Non-Docker startup (vLLM + bootstrap + app)
├── config.yaml           # Model, embedding, RAG, Discord settings
├── requirements.txt      # Python dependencies
├── app/
│   ├── main.py           # FastAPI + Discord bot entry point
│   ├── rag.py            # RAG engine (translate -> embed -> retrieve -> generate)
│   ├── discord_bot.py    # Discord bot with !ask command
│   ├── generate.py       # vLLM client (generate + translate_to_english)
│   ├── embeddings.py     # sentence-transformers wrapper
│   ├── retrieve.py       # ChromaDB client
│   ├── ingest.py         # Document ingestion pipeline
│   ├── chainlit_app.py   # Chainlit web UI (test mode)
│   └── config.py         # YAML config loader + env var overrides
├── scripts/
│   ├── bootstrap.py      # Fetch + ingest Python docs into ChromaDB
│   └── install.sh        # Host install script
└── data/                 # (generated) ChromaDB storage
```

### Data Flow

1. Student asks in Discord: `!ask hvad er en funktion i python?`
2. Discord bot (`app/discord_bot.py`) receives the question
3. RAG engine (`app/rag.py`):
   a. **Translate** — `Generator.translate_to_english()` sends the question to
      vLLM with a translation prompt (temperature 0). Returns English text.
   b. **Embed** — `Embedder.embed_query()` converts the English query to a
      vector using BAAI/bge-small-en-v1.5.
   c. **Retrieve** — `Retriever.query()` searches ChromaDB for the top-5
      matching documentation chunks. Each chunk includes title + URL in the
      context string.
   d. **Generate** — `Generator.generate()` sends the original question + the
      retrieved context to vLLM. The system prompt instructs the model to
      answer in the question's language, be pedagogical, cite specific
      sections, and skip thinking tokens.
4. Answer + top 2 sources returned to Discord

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Discord | discord.py 2.4.0+ | Bot framework |
| LLM | vLLM 0.28.0 + Qwen/Qwen3-8B | Answer generation + translation |
| Embeddings | sentence-transformers + BAAI/bge-small-en-v1.5 | Query/chunk embedding |
| Vector DB | ChromaDB 0.5.0+ | Document storage & retrieval |
| API | FastAPI + Uvicorn | REST endpoint (port 8001) |
| HTML Parsing | BeautifulSoup4 | Python docs parsing |
| Config | YAML + .env | Configuration |

---

## Current Configuration

### config.yaml (key settings)

```yaml
generator:
  model: "Qwen/Qwen3-8B"
  vllm_host: "localhost"
  vllm_port: 8000
  temperature: 0.3
  max_tokens: 1024
  system_prompt: |
    # Pedagogical instructions, language matching, citation format,
    # no thinking tokens. See config.yaml for full text.

embedding:
  model: "BAAI/bge-small-en-v1.5"
  device: "cpu"

rag:
  top_k: 5
  chunk_size: 512
  chunk_overlap: 64

chroma:
  persist_path: "/tmp/chroma"
  collection_name: "python_docs"

discord:
  enabled: true
  token: "<bot token>"
  prefix: "!"
  channel_id: ""
```

### vLLM Launch Flags (in start_rag.sh)

```
--model Qwen/Qwen3-8B
--port 8000
--gpu-memory-utilization 0.90
--max-model-len 8192
--enforce-eager              # Skip torch AOT compilation (avoids nvcc dependency)
--reasoning-parser qwen3     # Strip Qwen3 thinking tokens into separate field
```

### Environment Variables (set in start_rag.sh)

```bash
CUDA_HOME=/home/ucloud/.local/lib/python3.12/site-packages/nvidia/cu13
PATH="$CUDA_HOME/bin:$PATH"
FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
```

### Hardware

- **GPU:** NVIDIA B200 (Blackwell, sm100)
- **VRAM:** 183 GB total, ~15 GB model + ~142 GB KV cache
- **Model:** Qwen/Qwen3-8B (~15 GB in bf16)

---

## Debugging History (2026-09-09)

### 1. Repo Cleanup
- Removed 22 files + `.chainlit/` directory (old bot iterations, test files,
  mock modules, demo scripts, dev docs). Kept only the deployment-critical path.
- Tracked file `compute_budget.md` removed via `git rm`.

### 2. vLLM Not Installed (original failure)
- **Symptom:** `start_rag.sh` printed "Dependencies installed" then spent 10
  minutes polling `/health` with no server running.
- **Root cause:** `pip install ... | grep -q "Successfully"` swallowed the
  install failure. vLLM wasn't installed at all.
- **Fix:** Replaced the grep pipe with `if ! pip install ...; then exit 1`.
  Installed vLLM 0.28.0 manually.

### 3. Invalid Model ID
- **Symptom:** vLLM would have failed to download the model.
- **Root cause:** `start_rag.sh` hardcoded `Qwen/Qwen3-8B-27B` — not a valid
  HuggingFace ID. `config.yaml` had `Qwen/Qwen3-0.6B`.
- **Fix:** `start_rag.sh` now reads the model ID from `config.yaml`. Model
  changed to `Qwen/Qwen3-8B`.

### 4. FlashInfer JIT Compilation Failures (B200 / Blackwell)
Four nested issues, all in the FlashInfer attention/sampling kernel JIT build:

#### 4a. No nvcc found
- **Error:** `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`
- **Fix:** Set `CUDA_HOME` to the pip-installed CUDA toolkit at
  `/home/ucloud/.local/lib/python3.12/site-packages/nvidia/cu13` and added
  its `bin` to `PATH`.

#### 4b. CCCL version check failed
- **Error:** `CUDA compiler and CUDA toolkit headers are incompatible`
- **Root cause:** pip `nvcc` is 13.3 but the runtime headers are 13.0
  (`CUDART_VERSION 13000`). The bundled CCCL check rejects this mismatch.
- **Fix:** Set `FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"`
  to skip the version check.

#### 4c. Linker could not find libcudart.so
- **Error:** `/usr/bin/ld: cannot find -lcudart: No such file or directory`
- **Root cause:** pip CUDA uses `lib/` not `lib64/`, and only has the
  versioned `libcudart.so.13` (no unversioned symlink).
- **Fix:** Created symlinks:
  - `lib64` -> `lib`
  - `libcudart.so` -> `libcudart.so.13`
  - `stubs/libcuda.so` -> `/usr/lib64/libcuda.so`

#### 4d. Torch AOT compilation also needs nvcc
- **Fix:** Added `--enforce-eager` to skip torch AOT compilation entirely.

### 5. ModuleNotFoundError: No module named 'app'
- **Symptom:** `python app/main.py` crashed on `from app.config import load_config`.
- **Root cause:** Running `python app/main.py` doesn't put the project root
  on `sys.path`.
- **Fix:** Changed to `python -m app.main` in `start_rag.sh`.

### 6. Discord Bot Showed Offline
- **Symptom:** Bot appeared offline in Discord after `start_rag.sh`.
- **Root cause:** No Discord token — `config.yaml` had `token: ""` and
  `DISCORD_TOKEN` env var was not set. `app/main.py` skipped the bot and
  ran FastAPI-only.
- **Fix:** Set the bot token in `config.yaml` under `discord.token`.

### 7. Old Bot Processes Responding in Discord
- **Symptom:** Bot gave "Check the official Python documentation" answers
  even after running the updated `start_rag.sh`.
- **Root cause:** Three old standalone bot processes (`test_bot_final.py`,
  `rag_tutor_bot.py` x2) were still running in memory from earlier sessions.
  They connected to Discord with the old token and gave mock/hardcoded
  responses.
- **Fix:** Killed all three old processes (PIDs 17660, 27907, 31404).

### 8. Danish Questions Retrieved Wrong Chunks
- **Symptom:** Danish questions like "Hvordan bruger man en for-løkke?" got
  irrelevant chunks (Index pages) or refusals.
- **Root cause:** The embedding model (BAAI/bge-small-en-v1.5) is English-only.
  Danish queries produced poor embeddings that didn't match the English docs.
- **Fix:** Added `Generator.translate_to_english()` — sends the question to
  vLLM with a translation prompt (temperature 0) before retrieval. The
  English translation is used for embedding; the original Danish question is
  still sent to the generator for the final answer.

### 9. Qwen3 Thinking Tokens in Answers
- **Symptom:** Answers included internal reasoning ("Let me check the
  provided context...") before the actual answer.
- **Root cause:** Qwen3-8B emits thinking tokens by default.
- **Fix:** Added `--reasoning-parser qwen3` to the vLLM launch. This separates
  thinking tokens into a `reasoning_content` field that the app ignores.

### 10. English Questions Answered in Danish
- **Symptom:** English questions got Danish answers because the system prompt
  mentioned "Danish master's students".
- **Fix:** Tightened the language rule: "Detect the question's language from
  the words and grammar, not from this system prompt."

### 11. Too Many Sources in Discord
- **Symptom:** 5 sources made the Discord response too long.
- **Fix:** Changed `sources[:5]` to `sources[:2]` in `app/discord_bot.py`.

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-09-09 | Removed 22 obsolete files + .chainlit/ dir | User + Mistral Vibe |
| 2026-09-09 | Fixed vLLM install (fail loudly on pip error) | Mistral Vibe |
| 2026-09-09 | Fixed model ID: Qwen/Qwen3-8B (was Qwen3-8B-27B) | Mistral Vibe |
| 2026-09-09 | Set CUDA_HOME for pip CUDA 13 toolkit | Mistral Vibe |
| 2026-09-09 | Fixed FlashInfer CCCL check + libcudart symlinks | Mistral Vibe |
| 2026-09-09 | Added --enforce-eager to vLLM launch | Mistral Vibe |
| 2026-09-09 | Fixed app import: python -m app.main | Mistral Vibe |
| 2026-09-09 | Set Discord token in config.yaml | User + Mistral Vibe |
| 2026-09-09 | Killed old bot processes (test_bot_final, rag_tutor_bot) | Mistral Vibe |
| 2026-09-09 | Added Danish-to-English translation for retrieval | Mistral Vibe |
| 2026-09-09 | Added --reasoning-parser qwen3 (strip thinking tokens) | Mistral Vibe |
| 2026-09-09 | Fixed English questions getting Danish answers | Mistral Vibe |
| 2026-09-09 | Limited Discord sources to top 2 | User + Mistral Vibe |
| 2026-09-09 | Added pedagogical instructions to system prompt | User + Mistral Vibe |
| 2026-09-09 | Created get_started.md guide | Mistral Vibe |

---

*Generated by Mistral Vibe for AAUBot-Python project*
