"""ChromaDB vector store — embedded persistent client."""
import chromadb


class Retriever:
    def __init__(self, persist_path: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], embeddings: list, documents: list[str], metadatas: list[dict]):
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def query(self, query_embedding: list[float], top_k: int = 5) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()
