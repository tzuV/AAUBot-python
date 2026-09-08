"""Embedding model wrapper using sentence-transformers."""
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", normalize: bool = True):
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts, normalize_embeddings=self.normalize, show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            [text], normalize_embeddings=self.normalize, show_progress_bar=False
        )
        return embedding[0].tolist()
