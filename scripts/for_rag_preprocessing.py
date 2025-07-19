import os
import json
import re
from tqdm.auto import tqdm
from src.configs.config import RAGIndexArgs
from src.test.retriever import Retriever
from src.utils.print_utils import printi, printw

def convert_numpy_types(obj):
    """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def extract_multiple_choice_options(question_text):
    """
    단답형 문제에서 보기를 추출하는 함수

    Args:
        question_text: 질문 텍스트

    Returns:
        tuple: (질문_본문, 보기_리스트)
    """
    # 보기 패턴 매칭 (1. 2. 3. 4. 또는 ① ② ③ ④ 등)
    option_patterns = [
        r'(\d+)\.\s*([^\n\d]+?)(?=\s*\d+\.|$)',  # 1. 보기내용 2. 보기내용 형태
        r'([①②③④⑤⑥⑦⑧⑨⑩])\s*([^\n①-⑩]+?)(?=\s*[①-⑩]|$)',  # ① 보기내용 ② 보기내용 형태
        r'([가나다라마바사아자차카타파하])\.\s*([^\n가-하]+?)(?=\s*[가-하]\.|$)',  # 가. 보기내용 나. 보기내용 형태
    ]

    question_body = question_text
    options = []

    for pattern in option_patterns:
        matches = re.findall(pattern, question_text, re.MULTILINE | re.DOTALL)
        if matches:
            # 보기가 발견되면 질문 본문과 보기를 분리
            # 첫 번째 보기가 시작되는 지점을 찾아서 그 이전까지를 질문 본문으로 설정
            first_option_match = re.search(pattern, question_text, re.MULTILINE | re.DOTALL)
            if first_option_match:
                question_body = question_text[:first_option_match.start()].strip()
                options = [match[1].strip() for match in matches]
                break

    return question_body, options


def process_rag_queries_to_contexts(
    original_data_dir: str,
    output_data_dir: str,
    rag_index_args: RAGIndexArgs,
    splits: list = ["train", "dev"],
    top_k_per_query: int = 2
):
    """
    기존 데이터셋의 rag_queries를 사용해 RAG 검색 결과를 retrieved_contexts로 추가
    단답형 문제의 경우 보기를 분리해서 각각 검색

    Args:
        original_data_dir: 원본 데이터셋 경로
        output_data_dir: RAG가 추가된 데이터셋을 저장할 경로
        rag_index_args: RAG 설정
        splits: 처리할 데이터 분할 리스트
        top_k_per_query: 각 rag_query당 검색할 문서 수
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

        # 각 샘플에 대해 RAG 검색 수행
        processed_data = []
        total_option_searches = 0

        for example in tqdm(data, desc=f"Adding retrieved contexts to {split}", unit="example"):
            question = example["input"].get("question", "")
            question_type = example["input"].get("question_type", "")

            # 모든 retrieved contexts 저장할 리스트
            all_retrieved_contexts = []

            if question:
                # 선다형 문제인 경우에만 보기 추출
                if question_type == "선다형":
                    question_body, options = extract_multiple_choice_options(question)
                else:
                    question_body, options = question, []

                print("question_body:", question_body, "options:", options)

                # 1. 질문 본문으로 검색 수행
                retrieved_docs = retriever.retrieve(question_body, top_k=top_k_per_query)

                for doc in retrieved_docs:
                    context = {
                        "text": doc["text"],
                        "title": doc.get("title", "unknown"),
                        "chunk_id": doc.get("chunk_id", -1),
                        "query": question_body,
                        "query_type": "question_body",
                        "score": doc.get("score", 0.0)
                    }
                    all_retrieved_contexts.append(context)

                # 2. 각 보기로도 개별 검색 수행 (선다형이고 보기가 있는 경우)
                if question_type == "선다형" and options:
                    total_option_searches += len(options)
                    # printi(f"Found {len(options)} options for multiple choice question: {question_body}...")

                    for i, option in enumerate(options, 1):
                        # 보기 텍스트만으로 검색
                        option_retrieved_docs = retriever.retrieve(option, top_k=top_k_per_query)

                        for doc in option_retrieved_docs:
                            context = {
                                "text": doc["text"],
                                "title": doc.get("title", "unknown"),
                                "chunk_id": doc.get("chunk_id", -1),
                                "query": option,
                                "query_type": f"option_{i}",
                                "score": doc.get("score", 0.0)
                            }
                            all_retrieved_contexts.append(context)

            # 기존 데이터에 retrieved_contexts 추가
            example_with_contexts = example.copy()
            example_with_contexts["retrieved_contexts"] = all_retrieved_contexts

            # 디버깅을 위한 추가 정보 (선다형인 경우에만)
            if question_type == "선다형" and options:
                example_with_contexts["parsed_question_body"] = question_body
                example_with_contexts["parsed_options"] = options

            processed_data.append(example_with_contexts)

        # 저장
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(processed_data), f, ensure_ascii=False, indent=2)

        printi(f"Saved {split} dataset with retrieved contexts to {output_path}")

        # 통계 출력
        total_contexts = sum(len(ex.get("retrieved_contexts", [])) for ex in processed_data)
        avg_contexts_per_example = total_contexts / len(processed_data) if processed_data else 0
        multiple_choice_examples = sum(1 for ex in processed_data if ex["input"].get("question_type") == "선다형")
        examples_with_options = sum(1 for ex in processed_data if ex.get("parsed_options"))

        printi(f"  - Total examples: {len(processed_data)}")
        printi(f"  - Multiple choice examples: {multiple_choice_examples}")
        printi(f"  - Multiple choice examples with parsed options: {examples_with_options}")
        printi(f"  - Total option searches performed: {total_option_searches}")
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
            "chunk_size": rag_index_args.chunk_size,
            "chunk_overlap": rag_index_args.chunk_overlap,
            "multiple_choice_processing": "enabled (only for question_type='선다형')",
            "search_strategy": "question_body + individual_options (for 선다형 only)"
        }, f, ensure_ascii=False, indent=2)

    printi(f"RAG configuration saved to {config_path}")


if __name__ == "__main__":
    from src.configs.config import RAGIndexArgs

    # RAG 설정 로드
    rag_index_args = RAGIndexArgs()

    # 원본 데이터 경로
    original_dir = "datasets/merged_dataset_no_aug_v1-3-cot_remove_duplication"

    # RAG가 추가된 데이터를 저장할 경로
    output_dir = f"{original_dir}_for_rag"

    # 처리 실행
    process_rag_queries_to_contexts(
        original_data_dir=original_dir,
        output_data_dir=output_dir,
        rag_index_args=rag_index_args,
        splits=["train", "dev", "test"],
        top_k_per_query=2  # 질문본문 2개 + 각 보기별 2개씩
    )
