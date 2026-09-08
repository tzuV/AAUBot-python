"""FastAPI Q&A endpoint + Discord bot, started in one process."""
import asyncio

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from app.config import load_config
from app.rag import RAGEngine
from app.discord_bot import run_discord_bot

config = load_config("config.yaml")

app = FastAPI(
    title="AAUBot-Python",
    description="RAG Q&A bot for Python programming — powered by vLLM + ChromaDB",
)
rag_engine: RAGEngine | None = None


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.on_event("startup")
async def startup():
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine(config)


@app.get("/")
async def root():
    return {"name": "AAUBot-Python", "docs": "/docs", "ask": "POST /ask"}


@app.get("/health")
async def health():
    vllm_ready = False
    if rag_engine is not None:
        vllm_ready = await rag_engine.generator.is_ready()
    return {"status": "ok", "vllm_ready": vllm_ready}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    result = await rag_engine.ask(req.question)
    return AskResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
    )


async def main():
    global rag_engine
    rag_engine = RAGEngine(config)

    api_cfg = config["api"]
    server_config = uvicorn.Config(
        app,
        host=api_cfg["host"],
        port=api_cfg["port"],
        log_level="info",
    )
    server = uvicorn.Server(server_config)

    discord_cfg = config["discord"]
    if discord_cfg.get("enabled") and discord_cfg.get("token"):
        print("Starting Discord bot...")
        await asyncio.gather(
            server.serve(),
            run_discord_bot(rag_engine, discord_cfg),
        )
    else:
        print("Discord bot disabled. Set discord.enabled=true and discord.token in config.yaml, or set DISCORD_TOKEN env var.")
        await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
