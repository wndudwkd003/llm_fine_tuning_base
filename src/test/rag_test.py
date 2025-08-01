import os, json
from src.configs.config import RAGIndexArgs
from src.test.retriever import Retriever


def test_retrieve(query: str, top_k: int = 5):
    rag_index_args = RAGIndexArgs()
    retriever = Retriever(rag_index_args)

    print(f"\n[질의] {query}")
    retrieved = retriever.retrieve(query, top_k=top_k)

    print(f"\n[상위 {top_k}개 배경 문서]:\n")
    for i, r in enumerate(retrieved):
        print(f"[{i+1}] 제목: {r['title']}")
        print(f"내용: {r['text']}\n{'-'*80}\n")


if __name__ == "__main__":
    query = input("질문을 입력하세요: ").strip()
    test_retrieve(query)
