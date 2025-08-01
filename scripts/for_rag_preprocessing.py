import os
import json
import re
from collections import Counter
from tqdm.auto import tqdm
from src.configs.config import RAGIndexArgs
from src.test.retriever import Retriever
from src.utils.print_utils import printi, printw
import argparse

# BM25 라이브러리 임포트
try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    print("Warning: rank-bm25 not installed. Install with: pip install rank-bm25")
    HAS_RANK_BM25 = False


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
    """선다형 문제에서 보기를 추출하는 함수"""
    option_patterns = [
        r'(\d+)\.\s*([^\n\d]+?)(?=\s*\d+\.|$)',
        r'([①②③④⑤⑥⑦⑧⑨⑩])\s*([^\n①-⑩]+?)(?=\s*[①-⑩]|$)',
        r'([가나다라마바사아자차카타파하])\.\s*([^\n가-하]+?)(?=\s*[가-하]\.|$)',
    ]

    question_body = question_text
    options = []

    for pattern in option_patterns:
        matches = re.findall(pattern, question_text, re.MULTILINE | re.DOTALL)
        if matches:
            first_option_match = re.search(pattern, question_text, re.MULTILINE | re.DOTALL)
            if first_option_match:
                question_body = question_text[:first_option_match.start()].strip()
                options = [match[1].strip() for match in matches]
                break

    return question_body, options


def check_repeated_words(text, max_repeats=5):
    """텍스트에서 동일한 단어가 5번 이상 반복되는지 확인"""
    cleaned_text = re.sub(r'[^\w\s가-힣]', '', text)
    words = cleaned_text.split()
    word_counts = Counter(words)

    for count in word_counts.values():
        if count >= max_repeats:
            return True
    return False


def tokenize_korean(text):
    """간단한 한국어 토크나이저"""
    # 특수문자 제거하고 공백으로 분리
    cleaned_text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
    tokens = [token.strip() for token in cleaned_text.split() if len(token.strip()) > 1]
    return tokens


class BM25Retriever:
    """rank-bm25 라이브러리를 사용한 BM25 검색기"""

    def __init__(self, documents):
        """
        Args:
            documents: 문서 리스트 (각 문서는 {"text": str, "title": str, ...} 형태)
        """
        if not HAS_RANK_BM25:
            raise ImportError("rank-bm25 library is required. Install with: pip install rank-bm25")

        self.documents = documents
        printi("Building BM25 index with rank-bm25...")
        self._build_index()
        printi(f"BM25 index built for {len(self.documents)} documents")

    def _build_index(self):
        """BM25 인덱스 구축"""
        # 각 문서 토크나이즈
        self.tokenized_docs = []

        for doc in tqdm(self.documents, desc="Tokenizing documents for BM25"):
            tokens = tokenize_korean(doc.get("text", ""))
            self.tokenized_docs.append(tokens)

        # BM25 인덱스 생성
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query, top_k=5):
        """BM25로 문서 검색"""
        query_tokens = tokenize_korean(query)

        if not query_tokens:
            return []

        # BM25 점수 계산
        doc_scores = self.bm25.get_scores(query_tokens)

        # 점수와 인덱스를 함께 정렬
        scored_docs = [(score, idx) for idx, score in enumerate(doc_scores)]
        scored_docs.sort(reverse=True)

        # 상위 k개 반환
        results = []
        for score, doc_idx in scored_docs[:top_k]:
            if score > 0:  # 점수가 0보다 큰 것만
                doc = self.documents[doc_idx].copy()
                doc["score"] = float(score)
                doc["chunk_id"] = doc_idx
                results.append(doc)

        return results


class TFIDFRetriever:
    """rank-bm25가 없을 때 대안: scikit-learn TF-IDF"""

    def __init__(self, documents):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            self.documents = documents
            self.vectorizer = TfidfVectorizer(
                tokenizer=tokenize_korean,
                lowercase=False,
                token_pattern=None
            )

            printi("Building TF-IDF index as BM25 alternative...")

            # 문서 텍스트 추출
            doc_texts = [doc.get("text", "") for doc in documents]

            # TF-IDF 벡터화
            self.doc_vectors = self.vectorizer.fit_transform(doc_texts)

            printi(f"TF-IDF index built for {len(self.documents)} documents")

        except ImportError:
            raise ImportError("Neither rank-bm25 nor scikit-learn is available")

    def retrieve(self, query, top_k=5):
        """TF-IDF로 문서 검색"""
        from sklearn.metrics.pairwise import cosine_similarity

        # 쿼리 벡터화
        query_vector = self.vectorizer.transform([query])

        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()

        # 상위 k개 인덱스 추출
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc = self.documents[idx].copy()
                doc["score"] = float(similarities[idx])
                doc["chunk_id"] = idx
                results.append(doc)

        return results


def create_bm25_retriever(documents):
    """BM25 검색기 생성 (라이브러리 우선순위에 따라)"""
    if HAS_RANK_BM25:
        return BM25Retriever(documents)
    else:
        printi("rank-bm25 not available, using TF-IDF as alternative")
        return TFIDFRetriever(documents)


def remove_duplicates(docs_list, key_field="text"):
    """문서 리스트에서 중복 제거 (첫 번째 등장 순서 유지)"""
    seen = set()
    unique_docs = []

    for doc in docs_list:
        text = doc.get(key_field, "").strip()
        if text and text not in seen:
            seen.add(text)
            unique_docs.append(doc)

    return unique_docs


def dense_rerank_candidates(query, candidates, dense_retriever, top_k=5):
    """
    BM25 후보들을 Dense 모델로 재랭킹

    Args:
        query: 검색 쿼리 (전체 질문)
        candidates: BM25로 찾은 후보 문서들
        dense_retriever: Dense 검색기
        top_k: 최종 반환할 문서 수

    Returns:
        재랭킹된 상위 k개 문서
    """
    if not candidates:
        return []

    # Dense 검색기로 전체 질문에 대해 검색
    dense_results = dense_retriever.retrieve(query, top_k=len(candidates)*2)

    # 후보 문서들과 Dense 검색 결과를 매칭
    reranked_docs = []
    candidate_dict = {doc["text"].strip(): doc for doc in candidates}

    # Dense 검색 결과 중에서 후보에 있는 것들만 선택 (Dense 점수 순서대로)
    for dense_doc in dense_results:
        dense_text = dense_doc["text"].strip()
        if dense_text in candidate_dict:
            # 원본 BM25 문서에 Dense 점수 추가
            reranked_doc = candidate_dict[dense_text].copy()
            reranked_doc["bm25_score"] = reranked_doc.get("score", 0.0)  # 원래 BM25 점수 보존
            reranked_doc["dense_score"] = dense_doc.get("score", 0.0)
            reranked_doc["score"] = dense_doc.get("score", 0.0)  # Dense 점수를 최종 점수로 사용
            reranked_doc["search_method"] = "bm25_dense_rerank"
            reranked_docs.append(reranked_doc)

            # 이미 선택된 문서는 제거
            del candidate_dict[dense_text]

            if len(reranked_docs) >= top_k:
                break

    # 만약 Dense에서 매칭되지 않은 후보가 있고 아직 top_k에 못 채웠다면 BM25 점수 순으로 추가
    if len(reranked_docs) < top_k:
        remaining_candidates = list(candidate_dict.values())
        remaining_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

        for doc in remaining_candidates:
            if len(reranked_docs) >= top_k:
                break
            doc["search_method"] = "bm25_only"
            doc["bm25_score"] = doc.get("score", 0.0)
            reranked_docs.append(doc)

    return reranked_docs[:top_k]


def bm25_search_multiple_choice(question, question_body, options, bm25_retriever, candidates_per_query=30):
    """
    선다형 문제에 대한 BM25 검색

    Args:
        question: 전체 질문 텍스트
        question_body: 보기 제외한 질문 본문
        options: 보기 리스트
        bm25_retriever: BM25 검색기
        candidates_per_query: 각 검색당 수집할 후보 수

    Returns:
        모든 BM25 검색 결과를 합친 후보 리스트
    """
    all_candidates = []

    # 1. 전체 질문으로 검색
    full_question_docs = bm25_retriever.retrieve(question, top_k=candidates_per_query)
    for doc in full_question_docs:
        doc["bm25_query_type"] = "full_question"
    all_candidates.extend(full_question_docs)

    # 2. 보기 제외 질문만으로 검색
    if question_body and question_body != question:
        question_only_docs = bm25_retriever.retrieve(question_body, top_k=candidates_per_query)
        for doc in question_only_docs:
            doc["bm25_query_type"] = "question_only"
        all_candidates.extend(question_only_docs)

    # 3. 각 보기만으로 검색
    for i, option in enumerate(options, 1):
        option_docs = bm25_retriever.retrieve(option, top_k=candidates_per_query)
        for doc in option_docs:
            doc["bm25_query_type"] = f"option_{i}"
        all_candidates.extend(option_docs)

    # 4. 질문 본문 + 각 보기로 검색
    for i, option in enumerate(options, 1):
        combined_query = f"{question_body} {option}".strip()
        combined_docs = bm25_retriever.retrieve(combined_query, top_k=candidates_per_query)
        for doc in combined_docs:
            doc["bm25_query_type"] = f"question_with_option_{i}"
        all_candidates.extend(combined_docs)

    # 중복 제거 (BM25 점수가 높은 것 우선)
    unique_candidates = remove_duplicates(all_candidates)

    # BM25 점수 순으로 정렬
    unique_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

    return unique_candidates


def hybrid_search_strategy(question, question_type, bm25_retriever, dense_retriever,
                          candidates_per_query=30, final_top_k=5):
    """
    질문 유형에 따른 하이브리드 검색
    """

    if question_type == "선다형":
        # 선다형: 다양한 BM25 검색 → Dense 재랭킹
        question_body, options = extract_multiple_choice_options(question)

        if question_body and options:
            # 1단계: 다양한 BM25 검색으로 후보 수집
            bm25_candidates = bm25_search_multiple_choice(
                question, question_body, options, bm25_retriever, candidates_per_query
            )

            # 2단계: 전체 질문으로 Dense 재랭킹
            final_docs = dense_rerank_candidates(question, bm25_candidates, dense_retriever, final_top_k)

            # 디버깅 정보 추가
            for doc in final_docs:
                doc["parsed_question_body"] = question_body
                doc["parsed_options"] = options
                doc["num_bm25_candidates"] = len(bm25_candidates)

            return final_docs
        else:
            # 보기 파싱 실패시 일반 검색으로 처리
            question_type = "일반형"

    # 일반형: 단일 BM25 검색 → Dense 재랭킹
    bm25_candidates = bm25_retriever.retrieve(question, top_k=candidates_per_query * 4)  # 여유있게

    if not bm25_candidates:
        # BM25에서 아무것도 못 찾으면 Dense로 직접 검색
        dense_docs = dense_retriever.retrieve(question, top_k=final_top_k)
        for doc in dense_docs:
            doc["search_method"] = "dense_fallback"
        return dense_docs

    # Dense로 재랭킹
    final_docs = dense_rerank_candidates(question, bm25_candidates, dense_retriever, final_top_k)

    return final_docs


def filter_contexts(contexts, top_k=5, min_char_length=30):
    """검색된 컨텍스트들을 필터링하고 상위 k개만 반환"""
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

    # 상위 k개만 반환 (이미 정렬되어 있음)
    return filtered_contexts[:top_k]


def load_documents_for_bm25(rag_index_args):
    """BM25용 문서 로드"""
    meta_path = os.path.join(rag_index_args.index_dir, rag_index_args.meta_base)

    documents = []
    printi(f"Loading documents for BM25 from {meta_path}...")

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading documents"):
            doc = json.loads(line)
            documents.append({
                "text": doc.get("model_text", doc.get("text", "")),
                "title": doc.get("title", "unknown"),
                "corpus": doc.get("corpus", "unknown")
            })

    printi(f"Loaded {len(documents)} documents for BM25")
    return documents


def process_rag_queries_to_contexts(
    original_data_dir: str,
    output_data_dir: str,
    rag_index_args: RAGIndexArgs,
    splits: list = ["train", "dev"],
    top_k: int = 5,
    min_char_length: int = 30,
    candidates_per_query: int = 30
):
    """
    BM25 + Dense 하이브리드 검색을 사용해 retrieved_contexts 추가
    """

    printi("Initializing BM25 + Dense Hybrid Retriever...")

    # 1. 문서 로드 (BM25용)
    documents = load_documents_for_bm25(rag_index_args)

    # 2. BM25 검색기 초기화 (라이브러리 자동 선택)
    bm25_retriever = create_bm25_retriever(documents)

    # 3. Dense 검색기 초기화
    dense_retriever = Retriever(rag_index_args)

    os.makedirs(output_data_dir, exist_ok=True)

    for split in splits:
        input_path = os.path.join(original_data_dir, f"{split}.json")
        output_path = os.path.join(output_data_dir, f"{split}.json")

        if not os.path.exists(input_path):
            printw(f"Skipping {split}: {input_path} not found")
            continue

        printi(f"Processing {split} dataset with BM25→Dense hybrid strategy...")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        processed_data = []
        search_stats = {
            "multiple_choice_examples": 0,
            "general_examples": 0,
            "total_bm25_searches": 0,
            "total_dense_reranks": 0,
            "total_original_contexts": 0,
            "total_filtered_contexts": 0,
            "parsed_options_success": 0
        }

        for example in tqdm(data, desc=f"Adding hybrid retrieved contexts to {split}", unit="example"):
            question = example["input"].get("question", "")
            question_type = example["input"].get("question_type", "")

            if not question:
                example_with_contexts = example.copy()
                example_with_contexts["retrieved_contexts"] = []
                processed_data.append(example_with_contexts)
                continue

            # 하이브리드 검색 수행
            retrieved_contexts = hybrid_search_strategy(
                question, question_type, bm25_retriever, dense_retriever,
                candidates_per_query, top_k
            )

            # 통계 업데이트
            if question_type == "선다형":
                search_stats["multiple_choice_examples"] += 1
                question_body, options = extract_multiple_choice_options(question)
                if question_body and options:
                    search_stats["total_bm25_searches"] += 2 + len(options) * 2  # 전체질문 + 질문만 + 각보기 + 질문+각보기
                    search_stats["parsed_options_success"] += 1
                else:
                    search_stats["total_bm25_searches"] += 1
            else:
                search_stats["general_examples"] += 1
                search_stats["total_bm25_searches"] += 1

            search_stats["total_dense_reranks"] += 1

            # 검색 결과를 표준 형태로 변환
            standardized_contexts = []
            for doc in retrieved_contexts:
                context = {
                    "text": doc["text"],
                    "title": doc.get("title", "unknown"),
                    "chunk_id": doc.get("chunk_id", -1),
                    "score": doc.get("score", 0.0),
                    "search_method": doc.get("search_method", "hybrid"),
                    "bm25_query_type": doc.get("bm25_query_type", "unknown")
                }

                # 추가 점수 정보
                if "dense_score" in doc:
                    context["dense_score"] = doc["dense_score"]
                if "bm25_score" in doc:
                    context["bm25_score"] = doc["bm25_score"]

                standardized_contexts.append(context)

            # 필터링 적용
            original_count = len(standardized_contexts)
            filtered_contexts = filter_contexts(standardized_contexts, top_k, min_char_length)
            filtered_count = len(filtered_contexts)

            search_stats["total_original_contexts"] += original_count
            search_stats["total_filtered_contexts"] += filtered_count

            # 결과 저장
            example_with_contexts = example.copy()
            example_with_contexts["retrieved_contexts"] = filtered_contexts

            # 디버깅 정보 추가 (선다형인 경우)
            if question_type == "선다형" and retrieved_contexts:
                first_doc = retrieved_contexts[0]
                if "parsed_question_body" in first_doc:
                    example_with_contexts["parsed_question_body"] = first_doc["parsed_question_body"]
                    example_with_contexts["parsed_options"] = first_doc["parsed_options"]

            processed_data.append(example_with_contexts)

        # 저장
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(processed_data), f, ensure_ascii=False, indent=2)

        printi(f"Saved {split} dataset with hybrid retrieved contexts to {output_path}")

        # 통계 출력
        total_contexts = sum(len(ex.get("retrieved_contexts", [])) for ex in processed_data)
        avg_contexts_per_example = total_contexts / len(processed_data) if processed_data else 0

        # 검색 방법별 통계
        method_stats = {"hybrid": 0, "bm25_only": 0, "dense_fallback": 0}
        bm25_query_stats = {}

        for ex in processed_data:
            for ctx in ex.get("retrieved_contexts", []):
                method = ctx.get("search_method", "unknown")
                if method in method_stats:
                    method_stats[method] += 1

                bm25_query_type = ctx.get("bm25_query_type", "unknown")
                bm25_query_stats[bm25_query_type] = bm25_query_stats.get(bm25_query_type, 0) + 1

        printi(f"  - Total examples: {len(processed_data)}")
        printi(f"  - Multiple choice examples: {search_stats['multiple_choice_examples']}")
        printi(f"  - General examples: {search_stats['general_examples']}")
        printi(f"  - Parsed options success: {search_stats['parsed_options_success']}")
        printi(f"  - Total BM25 searches: {search_stats['total_bm25_searches']}")
        printi(f"  - Total Dense reranks: {search_stats['total_dense_reranks']}")
        printi(f"  - Contexts by method: {method_stats}")
        printi(f"  - BM25 query types: {bm25_query_stats}")
        printi(f"  - Original contexts: {search_stats['total_original_contexts']}")
        printi(f"  - Filtered contexts: {search_stats['total_filtered_contexts']}")
        printi(f"  - Average contexts per example: {avg_contexts_per_example:.1f}")

    # 설정 파일 저장
    config_path = os.path.join(output_data_dir, "bm25_dense_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "index_dir": rag_index_args.index_dir,
            "search_strategy": "BM25_to_Dense_Hybrid",
            "bm25_library": "rank-bm25" if HAS_RANK_BM25 else "scikit-learn TF-IDF",
            "top_k": top_k,
            "candidates_per_query": candidates_per_query,
            "min_char_length": min_char_length,
            "multiple_choice_strategy": {
                "bm25_searches": [
                    "full_question (전체 질문)",
                    "question_only (보기 제외 질문)",
                    "option_N (각 보기)",
                    "question_with_option_N (질문 + 각 보기)"
                ],
                "dense_rerank": "full_question (전체 질문으로 재랭킹)"
            },
            "general_strategy": {
                "bm25_search": "full_question",
                "dense_rerank": "full_question"
            },
            "filtering_rules": [
                f"minimum {min_char_length} characters (excluding spaces)",
                "remove texts with same word repeated 5+ times"
            ]
        }, f, ensure_ascii=False, indent=2)

    printi(f"BM25+Dense hybrid configuration saved to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RAG queries using BM25→Dense hybrid search")
    parser.add_argument("--original_dir", type=str,
                       default="datasets/merged_dataset_no_aug_v1-3_remove_duplication",
                       help="Original data directory")
    parser.add_argument("--candidates_per_query", type=int, default=30,
                       help="Number of candidates per BM25 query")

    args = parser.parse_args()

    # RAG 설정
    rag_index_args = RAGIndexArgs()
    rag_index_args.index_dir = "rag_index/kowikitext"

    output_dir = f"{args.original_dir}_for_rag_v2"

    # 처리 실행
    process_rag_queries_to_contexts(
        original_data_dir=args.original_dir,
        output_data_dir=output_dir,
        rag_index_args=rag_index_args,
        splits=["train", "dev", "test"],
        top_k=5,
        min_char_length=30,
        candidates_per_query=args.candidates_per_query
    )
