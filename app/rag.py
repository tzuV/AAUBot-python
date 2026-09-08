"""RAG engine — ties embedder, retriever, and generator together."""
from app.embeddings import Embedder
from app.retrieve import Retriever
from app.generate import Generator


class RAGEngine:
    def __init__(self, config: dict):
        gen_cfg = config["generator"]
        emb_cfg = config["embedding"]
        rag_cfg = config["rag"]
        chroma_cfg = config["chroma"]

        self.embedder = Embedder(
            model_name=emb_cfg["model"],
            device=emb_cfg.get("device", "cpu"),
            normalize=emb_cfg.get("normalize", True),
        )
        self.retriever = Retriever(
            persist_path=chroma_cfg["persist_path"],
            collection_name=chroma_cfg["collection_name"],
        )
        self.generator = Generator(
            model=gen_cfg["model"],
            vllm_host=gen_cfg["vllm_host"],
            vllm_port=gen_cfg["vllm_port"],
            temperature=gen_cfg["temperature"],
            max_tokens=gen_cfg["max_tokens"],
            top_p=gen_cfg["top_p"],
            system_prompt=gen_cfg["system_prompt"],
        )
        self.top_k = rag_cfg["top_k"]

    async def ask(self, question: str) -> dict:
        # 1. Embed the question
        query_embedding = self.embedder.embed_query(question)

        # 2. Retrieve top-k chunks
        results = self.retriever.query(query_embedding, top_k=self.top_k)

        # 3. Build context string from retrieved chunks
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        context_parts = []
        sources = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            title = meta.get("title", "Unknown")
            context_parts.append(f"[{i + 1}] {title}\n{doc}")
            sources.append({"title": title, "url": meta.get("url", "")})
        context = "\n\n".join(context_parts)

        # 4. Generate answer via vLLM
        answer = await self.generator.generate(question, context)

        return {"answer": answer, "sources": sources}
