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
        device: str = "cuda",
        # use_query_preprocessing: bool = True
    ):
        printi("Initializing Retriever")
        self.device = device
        # self.use_query_preprocessing = use_query_preprocessing


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
        # 쿼리 완전 전처리 (검색용)
        # if self.use_query_preprocessing:
        #     # processed_query = advanced_preprocess_text(query, use_morphological_analysis=True)
        #     # if not processed_query:
        #     #     processed_query = query  # 전처리 실패시 원본 사용
        #     pass
        # else:
        processed_query = query  # 전처리 없이 원본 사용

        query_embedding = self.embedding_model.encode(
            [processed_query],  # 전처리된 쿼리 사용
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device
        )

        query_embedding = query_embedding.astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1:
                meta = self.chunks[idx]
                results.append({
                    "text": meta.get("model_text", meta.get("text", "")),
                    "corpus": meta.get("corpus", "unknown"),
                    "title": meta.get("title", "unknown"),
                    "chunk_id": int(idx),  # 수정: int()로 변환
                    "score": float(distances[0][len(results)]) if 'distances' in locals() else 0.0  # 수정: float()로 변환
                })

        return results

    def batch_retrieve(self, queries: list[str], top_k: int = 5, batch_size: int = 32):
        """
        배치로 여러 쿼리를 처리하여 GPU 효율성 향상

        Args:
            queries: 검색할 쿼리 리스트
            top_k: 각 쿼리당 반환할 결과 수
            batch_size: 임베딩 배치 크기

        Returns:
            각 쿼리에 대한 검색 결과 리스트
        """
        if not queries:
            return []

        # 쿼리 전처리 (설정에 따라)
        processed_queries = []
        for query in queries:
            if self.use_query_preprocessing:
                processed_query = advanced_preprocess_text(query, use_morphological_analysis=True)
                if not processed_query:
                    processed_query = query
            else:
                processed_query = query
            processed_queries.append(processed_query)

        # 배치로 임베딩 생성
        printi(f"Generating embeddings for {len(processed_queries)} queries...")
        query_embeddings = self.embedding_model.encode(
            processed_queries,
            batch_size=batch_size,  # 배치 크기 설정
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
            show_progress_bar=True  # 진행률 표시
        )

        query_embeddings = query_embeddings.astype("float32")

        # 배치로 검색 수행
        printi(f"Performing batch search...")
        distances, indices = self.index.search(query_embeddings, top_k)

        # 결과 정리
        all_results = []
        for i in range(len(queries)):
            results = []
            for idx in indices[i]:
                if idx != -1:
                    meta = self.chunks[idx]
                    results.append({
                        "text": meta.get("model_text", meta.get("text", "")),
                        "corpus": meta.get("corpus", "unknown"),
                        "title": meta.get("title", "unknown"),
                        "chunk_id": idx,
                        "score": float(distances[i][len(results)])  # 검색 점수 추가
                    })
            all_results.append(results)

        return all_results
