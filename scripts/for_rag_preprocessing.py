import os
import json
from tqdm.auto import tqdm
from src.configs.config import RAGIndexArgs
from src.test.retriever import Retriever
from src.utils.print_utils import printi, printw

def preprocess_with_rag(
    original_data_dir: str,
    output_data_dir: str,
    rag_index_args: RAGIndexArgs,
    splits: list = ["train", "dev", "test"]
):
    """
    기존 데이터셋에 RAG 검색 결과를 미리 추가하여 새로운 데이터셋 생성

    Args:
        original_data_dir: 원본 데이터셋 경로
        output_data_dir: RAG가 추가된 데이터셋을 저장할 경로
        rag_index_args: RAG 설정
        splits: 처리할 데이터 분할 리스트
    """

    # Retriever 초기화
    printi("Initializing RAG Retriever...")
    retriever = Retriever(rag_index_args)

    # 출력 디렉토리 생성
    os.makedirs(output_data_dir, exist_ok=True)

    # 각 split 처리
    for split in splits:
        input_path = os.path.join(original_data_dir, f"{split}.json")
        output_path = os.path.join(output_data_dir, f"{split}.json")

        if not os.path.exists(input_path):
            printw(f"Skipping {split}: {input_path} not found")
            continue

        printi(f"Processing {split} dataset...")

        # 데이터 로드
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 각 샘플에 대해 RAG 검색 수행
        processed_data = []
        for example in tqdm(data, desc=f"Adding RAG to {split}", unit="example"):
            # 질문 추출
            question = example["input"]["question"]

            # RAG 검색
            retrieved_docs = retriever.retrieve(question, top_k=rag_index_args.top_k)

            # 검색 결과를 데이터에 추가
            example_with_rag = example.copy()
            example_with_rag["retrieved_contexts"] = [
                {
                    "text": doc["text"],
                    "source": doc.get("source", "unknown"),
                    "chunk_id": doc.get("chunk_id", -1)
                }
                for doc in retrieved_docs
            ]

            processed_data.append(example_with_rag)

        # 저장
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        printi(f"Saved {split} dataset with RAG to {output_path}")

    printi(f"All datasets processed and saved to {output_data_dir}")

    # 설정 파일 복사 (참고용)
    config_path = os.path.join(output_data_dir, "rag_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "index_dir": rag_index_args.index_dir,
            "model_name": rag_index_args.model_name,
            "top_k": rag_index_args.top_k,
            "chunk_size": rag_index_args.chunk_size,
            "chunk_overlap": rag_index_args.chunk_overlap
        }, f, ensure_ascii=False, indent=2)

    printi(f"RAG configuration saved to {config_path}")


if __name__ == "__main__":
    from src.configs.config import DataArgs, RAGIndexArgs

    # 설정 로드
    data_args = DataArgs()
    rag_index_args = RAGIndexArgs()

    # 원본 데이터 경로
    original_dir = data_args.data_dir

    # RAG가 추가된 데이터를 저장할 경로 (폴더명에 _for_rag 추가)
    output_dir = f"{original_dir}_for_rag"

    # 전처리 실행
    preprocess_with_rag(
        original_data_dir=original_dir,
        output_data_dir=output_dir,
        rag_index_args=rag_index_args,
        splits=["train", "dev", "test"]
    )
