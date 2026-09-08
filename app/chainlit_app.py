"""Chainlit browser UI for testing the RAG bot without Discord.

Run with: chainlit run app/chainlit_app.py --host 0.0.0.0 --port 8001
(start.sh does this automatically when test.enabled is true in config.yaml)
"""
import chainlit as cl

from app.config import load_config
from app.rag import RAGEngine

config = load_config("config.yaml")
rag_engine = RAGEngine(config)


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "AAUBot-Python is ready. Ask me a Python programming question "
            "and I'll answer using the official Python documentation."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    result = await rag_engine.ask(message.content)

    response = result["answer"]
    sources = result.get("sources", [])

    if sources:
        source_lines = "\n".join(
            f"- [{s['title']}]({s['url']})" for s in sources[:5]
        )
        response += "\n\n**Sources:**\n" + source_lines

    await cl.Message(content=response).send()
