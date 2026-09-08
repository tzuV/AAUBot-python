#!/usr/bin/env python3
"""Bootstrap: fetch docs, chunk, embed, and store in ChromaDB.
Idempotent — skips work that's already done.

Run standalone or as part of start.sh inside the Docker container.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_config
from app.embeddings import Embedder
from app.retrieve import Retriever
from app.ingest import ingest


def main():
    config = load_config("config.yaml")

    emb_cfg = config["embedding"]
    chroma_cfg = config["chroma"]
    rag_cfg = config["rag"]
    docs_cfg = config["docs"]

    print("=== AAUBot Bootstrap ===")
    print(f"  Embedding model: {emb_cfg['model']} on {emb_cfg.get('device', 'cpu')}")

    embedder = Embedder(
        model_name=emb_cfg["model"],
        device=emb_cfg.get("device", "cpu"),
        normalize=emb_cfg.get("normalize", True),
    )

    retriever = Retriever(
        persist_path=chroma_cfg["persist_path"],
        collection_name=chroma_cfg["collection_name"],
    )

    if retriever.count() > 0:
        print(f"  ChromaDB already has {retriever.count()} chunks. Skipping ingest.")
        return

    count = ingest(
        docs_url=docs_cfg["url"],
        cache_path=docs_cfg["cache_path"],
        embedder=embedder,
        retriever=retriever,
        chunk_size=rag_cfg["chunk_size"],
        chunk_overlap=rag_cfg["chunk_overlap"],
    )
    print(f"  Bootstrap complete: {count} chunks ready.")


if __name__ == "__main__":
    main()
