import os
import json
import re
from collections import Counter
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


def check_repeated_words(text, max_repeats=5):
    """
    텍스트에서 동일한 단어가 5번 이상 반복되는지 확인

    Args:
        text: 확인할 텍스트
        max_repeats: 허용되는 최대 반복 횟수

    Returns:
        bool: 5번 이상 반복되는 단어가 있으면 True
    """
    # 특수문자 제거하고 공백 기준으로 단어 분리
    cleaned_text = re.sub(r'[^\w\s가-힣]', '', text)
    words = cleaned_text.split()

    # 단어 빈도 계산
    word_counts = Counter(words)

    # 5번 이상 반복되는 단어가 있는지 확인
    for count in word_counts.values():
        if count >= max_repeats:
            return True

    return False


def filter_and_process_contexts(contexts, min_char_length=30):
    """
    검색된 컨텍스트들을 필터링하고 처리

    Args:
        contexts: 검색된 컨텍스트 리스트
        min_char_length: 최소 문자 길이 (공백 제외)

    Returns:
        필터링되고 정렬된 컨텍스트 리스트
    """
    filtered_contexts = []

    for context in contexts:
        text = context["text"].strip()

        # 공백 제거 후 길이 확인
        text_no_spaces = re.sub(r'\s+', '', text)
        if len(text_no_spaces) < min_char_length:
            continue

        # 동일한 단어 5번 이상 반복 확인
        if check_repeated_words(text):
            continue

        filtered_contexts.append(context)

    # 점수 순으로 정렬 (높은 점수부터)
    filtered_contexts.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # title 기준으로 최대 2개씩만 유지
    title_counts = {}
    final_contexts = []

    for context in filtered_contexts:
        title = context.get("title", "unknown")
        current_count = title_counts.get(title, 0)

        if current_count < 2:
            final_contexts.append(context)
            title_counts[title] = current_count + 1

    return final_contexts


def process_rag_queries_to_contexts(
    original_data_dir: str,
    output_data_dir: str,
    rag_index_args: RAGIndexArgs,
    splits: list = ["train", "dev"],
    top_k_per_search: int = 5,
    min_char_length: int = 30
):
    """
    기존 데이터셋의 rag_queries를 사용해 RAG 검색 결과를 retrieved_contexts로 추가
    5가지 검색 방식: 키워드만, 질문만, 질문+모든보기, 질문+각보기, 각보기만
    각각 5개씩 검색하고 필터링 적용

    Args:
        original_data_dir: 원본 데이터셋 경로
        output_data_dir: RAG가 추가된 데이터셋을 저장할 경로
        rag_index_args: RAG 설정
        splits: 처리할 데이터 분할 리스트
        top_k_per_search: 각 검색당 가져올 문서 수
        min_char_length: 최소 문자 길이 (공백 제외)
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
        search_stats = {
            "keyword_searches": 0,
            "question_searches": 0,
            "question_all_options_searches": 0,
            "question_each_option_searches": 0,
            "option_only_searches": 0,
            "total_original_contexts": 0,
            "total_filtered_contexts": 0
        }

        for example in tqdm(data, desc=f"Adding retrieved contexts to {split}", unit="example"):
            question = example["input"].get("question", "")
            question_type = example["input"].get("question_type", "")
            topic_keyword = example["input"].get("topic_keyword", "")

            # 모든 retrieved contexts 저장할 리스트
            all_retrieved_contexts = []

            # 1. 키워드만 검색 (topic_keyword가 있을 때)
            if topic_keyword and topic_keyword.strip():
                keyword_retrieved_docs = retriever.retrieve(topic_keyword, top_k=top_k_per_search)
                search_stats["keyword_searches"] += 1

                for doc in keyword_retrieved_docs:
                    context = {
                        "text": doc["text"],
                        "title": doc.get("title", "unknown"),
                        "chunk_id": doc.get("chunk_id", -1),
                        "query": topic_keyword,
                        "query_type": "keyword_only",
                        "score": doc.get("score", 0.0)
                    }
                    all_retrieved_contexts.append(context)

            # 2. 질문으로만 검색
            if question:
                if question_type == "선다형":
                    question_body, options = extract_multiple_choice_options(question)

                    # 질문 본문만으로 검색
                    if question_body:
                        retrieved_docs = retriever.retrieve(question_body, top_k=top_k_per_search)
                        search_stats["question_searches"] += 1

                        for doc in retrieved_docs:
                            context = {
                                "text": doc["text"],
                                "title": doc.get("title", "unknown"),
                                "chunk_id": doc.get("chunk_id", -1),
                                "query": question_body,
                                "query_type": "question_only",
                                "score": doc.get("score", 0.0)
                            }
                            all_retrieved_contexts.append(context)

                        # 3. 질문 + 모든 보기 (원본 그대로) 검색
                        if options:
                            full_question = question  # 원본 질문 전체
                            retrieved_docs = retriever.retrieve(full_question, top_k=top_k_per_search)
                            search_stats["question_all_options_searches"] += 1

                            for doc in retrieved_docs:
                                context = {
                                    "text": doc["text"],
                                    "title": doc.get("title", "unknown"),
                                    "chunk_id": doc.get("chunk_id", -1),
                                    "query": full_question,
                                    "query_type": "question_with_all_options",
                                    "score": doc.get("score", 0.0)
                                }
                                all_retrieved_contexts.append(context)

                            # 4. 질문 + 각 보기 검색
                            for i, option in enumerate(options, 1):
                                combined_query = f"{question_body} {option}"
                                retrieved_docs = retriever.retrieve(combined_query, top_k=top_k_per_search)
                                search_stats["question_each_option_searches"] += 1

                                for doc in retrieved_docs:
                                    context = {
                                        "text": doc["text"],
                                        "title": doc.get("title", "unknown"),
                                        "chunk_id": doc.get("chunk_id", -1),
                                        "query": combined_query,
                                        "query_type": f"question_with_option_{i}",
                                        "score": doc.get("score", 0.0)
                                    }
                                    all_retrieved_contexts.append(context)

                            # 5. 각 보기만 검색
                            for i, option in enumerate(options, 1):
                                retrieved_docs = retriever.retrieve(option, top_k=top_k_per_search)
                                search_stats["option_only_searches"] += 1

                                for doc in retrieved_docs:
                                    context = {
                                        "text": doc["text"],
                                        "title": doc.get("title", "unknown"),
                                        "chunk_id": doc.get("chunk_id", -1),
                                        "query": option,
                                        "query_type": f"option_only_{i}",
                                        "score": doc.get("score", 0.0)
                                    }
                                    all_retrieved_contexts.append(context)
                    else:
                        # 보기가 파싱되지 않은 경우 질문 전체로만 검색
                        retrieved_docs = retriever.retrieve(question, top_k=top_k_per_search)
                        search_stats["question_searches"] += 1

                        for doc in retrieved_docs:
                            context = {
                                "text": doc["text"],
                                "title": doc.get("title", "unknown"),
                                "chunk_id": doc.get("chunk_id", -1),
                                "query": question,
                                "query_type": "question_only",
                                "score": doc.get("score", 0.0)
                            }
                            all_retrieved_contexts.append(context)

                else:
                    # 선다형이 아닌 경우: 질문만 검색
                    retrieved_docs = retriever.retrieve(question, top_k=top_k_per_search)
                    search_stats["question_searches"] += 1

                    for doc in retrieved_docs:
                        context = {
                            "text": doc["text"],
                            "title": doc.get("title", "unknown"),
                            "chunk_id": doc.get("chunk_id", -1),
                            "query": question,
                            "query_type": "question_only",
                            "score": doc.get("score", 0.0)
                        }
                        all_retrieved_contexts.append(context)

            # 필터링 및 처리 적용
            original_count = len(all_retrieved_contexts)
            filtered_contexts = filter_and_process_contexts(all_retrieved_contexts, min_char_length)
            filtered_count = len(filtered_contexts)

            search_stats["total_original_contexts"] += original_count
            search_stats["total_filtered_contexts"] += filtered_count

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

        printi(f"  - Total examples: {len(processed_data)}")
        printi(f"  - Multiple choice examples: {multiple_choice_examples}")
        printi(f"  - Multiple choice examples with parsed options: {examples_with_options}")
        printi(f"  - Examples with topic_keyword: {examples_with_keywords}")
        printi(f"  - Keyword searches: {search_stats['keyword_searches']}")
        printi(f"  - Question-only searches: {search_stats['question_searches']}")
        printi(f"  - Question+all-options searches: {search_stats['question_all_options_searches']}")
        printi(f"  - Question+each-option searches: {search_stats['question_each_option_searches']}")
        printi(f"  - Option-only searches: {search_stats['option_only_searches']}")
        printi(f"  - Original retrieved contexts: {search_stats['total_original_contexts']}")
        printi(f"  - Filtered retrieved contexts: {search_stats['total_filtered_contexts']}")
        printi(f"  - Filtered out contexts: {search_stats['total_original_contexts'] - search_stats['total_filtered_contexts']}")
        printi(f"  - Average contexts per example: {avg_contexts_per_example:.1f}")

    printi(f"All datasets processed and saved to {output_data_dir}")

    # 설정 파일 저장 (참고용)
    config_path = os.path.join(output_data_dir, "rag_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "index_dir": rag_index_args.index_dir,
            "model_name": rag_index_args.model_name,
            "top_k_per_search": top_k_per_search,
            "min_char_length": min_char_length,
            "chunk_size": rag_index_args.chunk_size,
            "chunk_overlap": rag_index_args.chunk_overlap,
            "search_types": [
                "keyword_only (topic_keyword)",
                "question_only",
                "question_with_all_options (original question for 선다형)",
                "question_with_option_N (question + each option)",
                "option_only_N (each option only)"
            ],
            "filtering_rules": [
                f"minimum {min_char_length} characters (excluding spaces)",
                "remove texts with same word repeated 5+ times",
                "maximum 2 contexts per title",
                "sort by score (descending)"
            ]
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
        top_k_per_search=5,    # 각 검색당 5개씩
        min_char_length=30     # 공백 제외 30자 미만 제거
    )
