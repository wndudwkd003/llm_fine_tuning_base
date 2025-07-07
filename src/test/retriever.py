import os, json, faiss
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

from src.configs.config import (
    RAGIndexArgs
)
from src.utils.print_utils import printi


class Retriever:
    def __init__(
        self,
        rag_index_args: RAGIndexArgs,
        device: str = "cuda"
    ):

        printi("Initializing Retriever")
        self.device = device

        idx_path = os.path.join(rag_index_args.index_dir, rag_index_args.index_base)
        self.index = faiss.read_index(idx_path)
        printi(f"FAISS index loaded from {idx_path}. Total vectors: {self.index.ntotal}")

        meta_path = os.path.join(rag_index_args.index_dir, rag_index_args.meta_base)

        self.chunks = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading metadata", unit="lines"):
                self.chunks.append(json.loads(line))

        self.embedding_model = SentenceTransformer(
            rag_index_args.model_name,
            device=self.device
        )
        printi(f"Embedding model loaded: {rag_index_args.model_name}")

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device
        )

        query_embedding = query_embedding.astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            meta = self.chunks[idx]
            results.append(meta)

        return results
