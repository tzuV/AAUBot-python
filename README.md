# AAUBot-Python 

This is a bot that will be a helping teacher for a group of danish students in a course on a programming in Python course for a master in IT-management.

The bots task will be to provide simple Q&A related to programming and the bot will be built as a Retrieval-Augmented Generation (RAG) system and it will draw on the official python documentation.

The RAG will be based on a local deployment of a LLM and a local vector database. 

The stack is:
- vLLM — local LLM serving (continuous batching for high concurrency)
- ChromaDB — vector store (CPU-backed, low footprint)
- FastAPI — HTTP layer in front of vLLM
- Discord webhook — the bot's front door; students ask questions in Discord,
  the webhook forwards to the FastAPI Q&A endpoint and returns the answer
- A local embedding model (candidates: `bge-m3`, `e5-large-v2`,
  `bge-small-en-v1.5`; ~1–2 GB VRAM)

## End goal

Students communicate with the bot through Discord. A Discord webhook (or a
small bot bridge) forwards each question to the FastAPI endpoint, the RAG
system answers, and the response is posted back to the channel. The Python
docs are the knowledge base; the bot is a helping teacher, not a chatbot with
free-form internet knowledge.

## One-command bootstrap

The whole system must start with a single command. That command:

1. Pulls the LLM model from Hugging Face
2. Fetches the official Python documentation
3. Chunks, embeds, and ingests the docs into ChromaDB
4. Starts vLLM (LLM + embedding model loaded on the GPU)
5. Launches the FastAPI Q&A endpoint
6. (Optionally) registers the Discord webhook / starts the bot bridge

## Pipeline

1. Bootstrap: one command → model download → doc fetch → ingest → serve
2. Ingest: fetch `docs.python.org` → chunk → embed → store in ChromaDB
3. Retrieve: embed question → top-k similarity search
4. Generate: vLLM answers with retrieved context stitched into the prompt
5. Serve: FastAPI exposes a Q&A endpoint that the Discord webhook calls

## Model choice

The LLM is the main VRAM cost. Candidates are sized in `compute_budget.md`
across three tiers (3B / 7B / 11B) and dense vs. MoE architectures. The size
choice is driven by available GPU VRAM and the 70-concurrent-students target.

## Run

```bash
# ponytail: placeholder until the code exists
docker-compose up
```

## Test mode

Set `test.enabled: true` in `config.yaml` to skip Discord and launch a
Chainlit browser UI for local testing instead. The same `docker-compose up`
command then serves a chat interface at `http://<VM-IP>:8001` where you can
ask questions directly in the browser — no Discord token or webhook setup
needed. Set it back to `false` for production (FastAPI endpoint + Discord bot).

This single command downloads the model, fetches and ingests the docs, starts
vLLM and the embedding model, and exposes the FastAPI endpoint wired to the
Discord webhook.

## Constraints

- Single-GPU VM; vLLM, the LLM, and the embedding model share one GPU's VRAM.
- Target: ~70 concurrent students. No latency bound set yet — see
  `compute_budget.md` for the throughput/VRAM trade-offs that bound this.
- Knowledge base starts at the official Python docs; may expand later.
- One-command bootstrap must be idempotent: re-running skips already-ingested
  docs and an already-downloaded model rather than redoing the work. 



