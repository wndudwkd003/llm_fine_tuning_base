import os
import json
from tqdm.auto import tqdm
from src.configs.config import RAGIndexArgs
from src.test.retriever import Retriever
from src.utils.print_utils import printi, printw

def process_rag_queries_to_contexts(
    original_data_dir: str,
    output_data_dir: str,
    rag_index_args: RAGIndexArgs,
    splits: list = ["train", "dev"],  # test 제외 (기본값 변경)
    top_k_per_query: int = 2,  # 각 쿼리당 검색할 문서 수
    batch_size: int = 32  # 배치 크기 추가
):
    """
    기존 데이터셋의 rag_queries를 사용해 RAG 검색 결과를 retrieved_contexts로 추가 (배치 처리)

    Args:
        original_data_dir: 원본 데이터셋 경로
        output_data_dir: RAG가 추가된 데이터셋을 저장할 경로
        rag_index_args: RAG 설정
        splits: 처리할 데이터 분할 리스트
        top_k_per_query: 각 rag_query당 검색할 문서 수
        batch_size: 배치 처리 크기
    """

    # Retriever 초기화
    printi("Initializing RAG Retriever...")
    retriever = Retriever(rag_index_args, use_query_preprocessing=False)

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

        # 배치 처리를 위한 쿼리 수집
        all_queries = []
        query_to_example_map = []  # (example_idx, query_idx) 매핑

        printi(f"Collecting all queries for batch processing...")
        for example_idx, example in enumerate(data):
            rag_queries = example["output"].get("rag_queries", [])
            for query_idx, query in enumerate(rag_queries):
                all_queries.append(query)
                query_to_example_map.append((example_idx, query_idx))

        printi(f"Total queries to process: {len(all_queries)}")

        # 배치 검색 수행
        printi(f"Performing batch retrieval with batch_size={batch_size}...")
        all_search_results = retriever.batch_retrieve(all_queries, top_k=top_k_per_query, batch_size=batch_size)

        # 결과를 example별로 재구성
        printi(f"Organizing results by examples...")
        processed_data = []
        for example_idx, example in enumerate(tqdm(data, desc=f"Organizing results for {split}", unit="example")):
            # 이 example에 해당하는 검색 결과들 찾기
            all_retrieved_contexts = []

            for result_idx, (ex_idx, query_idx) in enumerate(query_to_example_map):
                if ex_idx == example_idx:
                    query = example["output"]["rag_queries"][query_idx]
                    retrieved_docs = all_search_results[result_idx]

                    # 검색 결과를 retrieved_contexts 형태로 변환
                    for doc in retrieved_docs:
                        context = {
                            "text": doc["text"],
                            "source": doc.get("source", "unknown"),
                            "chunk_id": doc.get("chunk_id", -1),
                            "query": query,  # 어떤 쿼리로 검색된 결과인지 추가
                            "score": doc.get("score", 0.0)  # 검색 점수 추가
                        }
                        all_retrieved_contexts.append(context)

            # 기존 데이터에 retrieved_contexts 추가
            example_with_contexts = example.copy()
            example_with_contexts["retrieved_contexts"] = all_retrieved_contexts

            processed_data.append(example_with_contexts)

        # 저장
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        printi(f"Saved {split} dataset with retrieved contexts to {output_path}")

        # 통계 출력
        total_contexts = sum(len(ex.get("retrieved_contexts", [])) for ex in processed_data)
        avg_contexts_per_example = total_contexts / len(processed_data) if processed_data else 0
        printi(f"  - Total examples: {len(processed_data)}")
        printi(f"  - Total retrieved contexts: {total_contexts}")
        printi(f"  - Average contexts per example: {avg_contexts_per_example:.1f}")

    printi(f"All datasets processed and saved to {output_data_dir}")

    # 설정 파일 저장 (참고용)
    config_path = os.path.join(output_data_dir, "rag_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "index_dir": rag_index_args.index_dir,
            "model_name": rag_index_args.model_name,
            "top_k_per_query": top_k_per_query,
            "batch_size": batch_size,
            "chunk_size": rag_index_args.chunk_size,
            "chunk_overlap": rag_index_args.chunk_overlap,
            "total_queries_per_example": "variable (based on rag_queries length)"
        }, f, ensure_ascii=False, indent=2)

    printi(f"RAG configuration saved to {config_path}")

if __name__ == "__main__":
    from src.configs.config import RAGIndexArgs

    # RAG 설정 로드
    rag_index_args = RAGIndexArgs()

    # 원본 데이터 경로
    original_dir = "datasets/merged_dataset_no_aug_v1-3_rag_queries_remove_duplication"

    # RAG가 추가된 데이터를 저장할 경로
    output_dir = f"{original_dir}_for_rag"

    # 처리 실행
    process_rag_queries_to_contexts(
        original_data_dir=original_dir,
        output_data_dir=output_dir,
        rag_index_args=rag_index_args,
        splits=["train", "dev"],  # dev.json과 train.json만 처리
        top_k_per_query=2,  # 각 쿼리당 2개씩 검색 (총 6개)
        batch_size=128  # 배치 크기
    )
