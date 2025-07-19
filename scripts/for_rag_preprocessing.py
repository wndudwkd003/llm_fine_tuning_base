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


def filter_and_deduplicate_contexts(contexts: list, min_text_length: int = 15) -> list:
    """
    검색된 컨텍스트들을 필터링하고 중복 제거

    Args:
        contexts: 검색된 컨텍스트 리스트
        min_text_length: 최소 텍스트 길이

    Returns:
        필터링된 컨텍스트 리스트
    """
    seen_texts = set()
    filtered_contexts = []

    for context in contexts:
        text = context["text"].strip()

        # 짧은 텍스트 필터링
        if len(text) <= min_text_length:
            continue

        # 중복 제거 (텍스트 기준)
        if text in seen_texts:
            continue

        seen_texts.add(text)
        filtered_contexts.append(context)

    return filtered_contexts


def process_rag_queries_to_contexts(
    original_data_dir: str,
    output_data_dir: str,
    rag_index_args: RAGIndexArgs,
    splits: list = ["train", "dev"],
    top_k_per_query: int = 10,
    min_text_length: int = 15,
    keyword_top_k: int = 2
):
    """
    기존 데이터셋의 rag_queries를 사용해 RAG 검색 결과를 retrieved_contexts로 추가
    선다형 문제의 경우 질문+보기 조합으로 검색, 나머지는 질문 전체로 검색
    topic_keyword가 있으면 무조건 키워드로도 검색하여 상위 2개 결과 포함
    검색 후 중복 제거 및 짧은 텍스트 필터링 적용

    Args:
        original_data_dir: 원본 데이터셋 경로
        output_data_dir: RAG가 추가된 데이터셋을 저장할 경로
        rag_index_args: RAG 설정
        splits: 처리할 데이터 분할 리스트
        top_k_per_query: 각 rag_query당 검색할 문서 수
        min_text_length: 최소 텍스트 길이 (이보다 짧은 텍스트는 필터링)
        keyword_top_k: topic_keyword 검색시 가져올 상위 결과 수
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
        total_keyword_searches = 0
        total_filtered_contexts = 0
        total_original_contexts = 0

        for example in tqdm(data, desc=f"Adding retrieved contexts to {split}", unit="example"):
            question = example["input"].get("question", "")
            question_type = example["input"].get("question_type", "")
            topic_keyword = example["input"].get("topic_keyword", "")

            # 모든 retrieved contexts 저장할 리스트
            all_retrieved_contexts = []

            # 1. topic_keyword가 있으면 무조건 키워드로 검색 (상위 2개)
            if topic_keyword and topic_keyword.strip():
                keyword_retrieved_docs = retriever.retrieve(topic_keyword, top_k=keyword_top_k)
                total_keyword_searches += 1

                for doc in keyword_retrieved_docs:
                    context = {
                        "text": doc["text"],
                        "title": doc.get("title", "unknown"),
                        "chunk_id": doc.get("chunk_id", -1),
                        "query": topic_keyword,
                        "query_type": "topic_keyword",
                        "score": doc.get("score", 0.0),
                        "is_keyword_result": True  # 키워드 검색 결과임을 표시
                    }
                    all_retrieved_contexts.append(context)

            # 2. 질문으로 검색
            if question:
                if question_type == "선다형":
                    # 선다형: 질문 본문과 보기 분리
                    question_body, options = extract_multiple_choice_options(question)

                    print("question_body:", question_body, "options:", options)

                    if options:
                        # 질문 + 각 보기 조합으로 검색
                        for i, option in enumerate(options, 1):
                            combined_query = f"{question_body} {option}"
                            retrieved_docs = retriever.retrieve(combined_query, top_k=top_k_per_query)
                            total_option_searches += 1

                            for doc in retrieved_docs:
                                context = {
                                    "text": doc["text"],
                                    "title": doc.get("title", "unknown"),
                                    "chunk_id": doc.get("chunk_id", -1),
                                    "query": combined_query,
                                    "query_type": f"question_with_option_{i}",
                                    "score": doc.get("score", 0.0),
                                    "is_keyword_result": False
                                }
                                all_retrieved_contexts.append(context)
                    else:
                        # 보기가 파싱되지 않은 경우 질문 전체로 검색
                        retrieved_docs = retriever.retrieve(question, top_k=top_k_per_query)

                        for doc in retrieved_docs:
                            context = {
                                "text": doc["text"],
                                "title": doc.get("title", "unknown"),
                                "chunk_id": doc.get("chunk_id", -1),
                                "query": question,
                                "query_type": "full_question",
                                "score": doc.get("score", 0.0),
                                "is_keyword_result": False
                            }
                            all_retrieved_contexts.append(context)

                else:
                    # 선다형이 아닌 경우: 질문 전체로 검색
                    retrieved_docs = retriever.retrieve(question, top_k=top_k_per_query)

                    for doc in retrieved_docs:
                        context = {
                            "text": doc["text"],
                            "title": doc.get("title", "unknown"),
                            "chunk_id": doc.get("chunk_id", -1),
                            "query": question,
                            "query_type": "full_question",
                            "score": doc.get("score", 0.0),
                            "is_keyword_result": False
                        }
                        all_retrieved_contexts.append(context)

            # 중복 제거 및 필터링 적용
            original_count = len(all_retrieved_contexts)
            filtered_contexts = filter_and_deduplicate_contexts(all_retrieved_contexts, min_text_length)
            filtered_count = len(filtered_contexts)

            total_original_contexts += original_count
            total_filtered_contexts += filtered_count

            # 기존 데이터에 필터링된 retrieved_contexts 추가
            example_with_contexts = example.copy()
            example_with_contexts["retrieved_contexts"] = filtered_contexts

            # 디버깅을 위한 추가 정보 (선다형인 경우에만)
            if question_type == "선다형":
                question_body, options = extract_multiple_choice_options(question)
                if options:
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
        examples_with_keywords = sum(1 for ex in processed_data if ex["input"].get("topic_keyword", "").strip())

        # 키워드 검색 결과 통계
        keyword_contexts = sum(len([ctx for ctx in ex.get("retrieved_contexts", []) if ctx.get("is_keyword_result", False)]) for ex in processed_data)

        printi(f"  - Total examples: {len(processed_data)}")
        printi(f"  - Multiple choice examples: {multiple_choice_examples}")
        printi(f"  - Multiple choice examples with parsed options: {examples_with_options}")
        printi(f"  - Examples with topic_keyword: {examples_with_keywords}")
        printi(f"  - Total option searches performed: {total_option_searches}")
        printi(f"  - Total keyword searches performed: {total_keyword_searches}")
        printi(f"  - Keyword search contexts retrieved: {keyword_contexts}")
        printi(f"  - Original retrieved contexts: {total_original_contexts}")
        printi(f"  - Filtered retrieved contexts: {total_filtered_contexts}")
        printi(f"  - Filtered out contexts: {total_original_contexts - total_filtered_contexts}")
        printi(f"  - Average contexts per example: {avg_contexts_per_example:.1f}")

    printi(f"All datasets processed and saved to {output_data_dir}")

    # 설정 파일 저장 (참고용)
    config_path = os.path.join(output_data_dir, "rag_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "index_dir": rag_index_args.index_dir,
            "model_name": rag_index_args.model_name,
            "top_k_per_query": top_k_per_query,
            "keyword_top_k": keyword_top_k,
            "min_text_length": min_text_length,
            "chunk_size": rag_index_args.chunk_size,
            "chunk_overlap": rag_index_args.chunk_overlap,
            "multiple_choice_processing": "enabled (question+option combination for question_type='선다형')",
            "keyword_processing": f"enabled (top {keyword_top_k} results for topic_keyword)",
            "search_strategy": "topic_keyword (mandatory if exists) + question+option_combination (for 선다형) / full_question (for others)",
            "filtering": "text deduplication + minimum length filtering applied"
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
        top_k_per_query=10,
        min_text_length=15,   # 15자 이하 텍스트 제거
        keyword_top_k=2       # topic_keyword 검색시 상위 2개 결과
    )
