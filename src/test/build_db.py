import os, re, json, faiss, numpy as np, gc
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.configs.config import RAGIndexArgs
from src.utils.print_utils import printi
from src.data.record import Metadata

# ko-ww-stopwords 라이브러리 import
try:
    from ko_ww_stopwords.stop_words import ko_ww_stop_words
    from ko_ww_stopwords.tools import is_stop_word
    STOPWORDS_AVAILABLE = True
    printi("✅ ko-ww-stopwords 라이브러리 로드 성공")
except ImportError:
    STOPWORDS_AVAILABLE = False
    printi("⚠️ ko-ww-stopwords 라이브러리가 설치되지 않음. pip install ko-ww-stopwords")

# 헤더만 잡음
hdr_pat = re.compile(r'^\s*=+\s*(.+?)\s*=+\s*$', re.MULTILINE)
bullet_pat = re.compile(r'^\s*[•\-\–·\*]\s+', re.MULTILINE)
blank_pat = re.compile(r'\n{2,}')


def estimate_token_count(text: str) -> int:
    """
    한국어 텍스트의 대략적인 토큰 수 추정
    한국어는 대략 1.2-1.5 문자당 1토큰 정도
    """
    # 공백 제거 후 문자 수 계산
    clean_text = re.sub(r'\s+', '', text)
    # 한국어는 문자당 약 0.8-1.0 토큰으로 추정
    estimated_tokens = len(clean_text) * 0.9
    return int(estimated_tokens)

def remove_stopwords(text: str) -> tuple[str, list[str]]:
    """
    ko-ww-stopwords를 사용한 불용어 제거
    Returns: (filtered_text, removed_stopwords)
    """
    if not STOPWORDS_AVAILABLE:
        return text, []  # 라이브러리가 없으면 원본 텍스트와 빈 리스트 반환

    words = text.split()
    filtered_words = []
    removed_stopwords = []

    for word in words:
        if is_stop_word(word):
            removed_stopwords.append(word)
        else:
            filtered_words.append(word)

    return ' '.join(filtered_words), removed_stopwords

def light_preprocess_text(text: str) -> tuple[str, list[str]]:
    """
    가벼운 전처리 - 문맥과 의미를 보존하면서 노이즈만 제거
    저장과 임베딩 모두 이 텍스트 사용
    Returns: (processed_text, removed_stopwords)
    """
    if not text or not text.strip():
        return "", []

    # 1. HTML 태그, URL, 이메일 제거
    text = re.sub(r'<[^>]+>', '', text)  # HTML 태그
    text = re.sub(r'https?://[^\s]+', '', text)  # URL
    text = re.sub(r'www\.[^\s]+', '', text)  # www 링크
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)  # 이메일

    # 2. 한자 제거 (중국어 한자, 일본어 한자, 한국어 한자 모두 포함)
    text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', '', text)  # 한자 제거

    # 3. 극단적인 특수문자만 제거 (의미있는 문장부호는 유지)
    text = re.sub(r'[^\w\s가-힣.,!?;:()\-\'""]', ' ', text)  # 기본 문장부호는 유지

    # 4. 불용어 제거 (새로 추가)
    text, removed_stopwords = remove_stopwords(text)

    # 5. 연속된 공백과 줄바꿈 정리
    text = re.sub(r'\s+', ' ', text)  # 연속 공백을 하나로
    text = text.strip()

    return text, removed_stopwords

def is_korean_text(text: str, korean_ratio_threshold: float = 0.5) -> bool:
    """
    텍스트가 한국어인지 판단 (임계값을 낮춤)
    """
    if not text.strip():
        return False

    # 공백 제거
    content_chars = text.replace(' ', '')

    if len(content_chars) < 5:
        return False

    # 순수 한글 문자만 계산
    korean_chars = len(re.findall(r'[가-힣]', content_chars))

    # 한국어 비율 계산
    korean_ratio = korean_chars / len(content_chars)

    return korean_ratio >= korean_ratio_threshold

def clean(txt: str) -> str:
    """기본 텍스트 정리"""
    # 줄 단위로 헤더 정제
    lines = txt.split('\n')
    lines = [hdr_pat.sub(r'[SECTION]\1', line) for line in lines]
    txt = '\n'.join(lines)

    # [SECTION]= = 제목 = = → [SECTION]제목  로 정리
    txt = re.sub(r'\[SECTION\][\s=]*([\w가-힣A-Za-z0-9 _\-()]+)[\s=]*', r'[SECTION]\1', txt)

    txt = bullet_pat.sub('', txt)
    txt = blank_pat.sub('\n', txt)
    return txt.strip()

def load_documents(path: str) -> list[tuple[str, str]]:
    """문서 로드"""
    documents = []
    current_title = None
    current_body = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # level-1 제목을 문서 시작 신호로 사용
            if re.match(r'^\s*=+\s*[^=]+?\s*=+\s*$', line):
                if current_title and current_body:
                    documents.append((current_title, "\n".join(current_body).strip()))
                    current_body = []
                current_title = re.sub(r'^\s*=+\s*|\s*=+\s*$', '', line).strip()
            else:
                current_body.append(line)

    if current_title and current_body:
        documents.append((current_title, "\n".join(current_body).strip()))

    return documents

def process_batch_and_save(chunks_batch, metas_batch, model, index, meta_file_handle, batch_size):
    """배치를 임베딩하고 바로 저장"""
    if not chunks_batch:
        return

    # 임베딩 생성 (전처리된 텍스트로)
    vecs = model.encode(
        chunks_batch,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # FAISS 인덱스에 추가
    index.add(np.float32(vecs))

    # 메타데이터 파일에 바로 저장
    for meta in metas_batch:
        meta_file_handle.write(json.dumps(meta.__dict__, ensure_ascii=False) + "\n")

    # 메모리 정리
    del vecs
    gc.collect()

    printi(f"Processed batch of {len(chunks_batch)} chunks")

def process_single_corpus(corpus_info: dict, rag_index_args: RAGIndexArgs, model, splitter,
                         korean_ratio_threshold: float) -> tuple:

    MIN_TOKEN_COUNT = 32

    """단일 코퍼스를 처리하고 개별 파일로 저장"""
    c_dir = corpus_info["dir"]
    c_name = os.path.basename(c_dir)
    c_base = corpus_info["base"]
    exts = corpus_info["ext"]

    # 개별 코퍼스용 디렉토리 생성
    corpus_index_dir = os.path.join(rag_index_args.index_dir, c_name)
    os.makedirs(corpus_index_dir, exist_ok=True)

    # 첫 번째 배치로 차원 확인을 위한 더미 임베딩
    dummy_vec = model.encode(["dummy"], normalize_embeddings=True)
    dim = dummy_vec.shape[1]

    # 개별 FAISS 인덱스 초기화
    corpus_index = faiss.IndexFlatIP(dim)

    # 개별 메타데이터 파일 경로
    corpus_meta_path = os.path.join(corpus_index_dir, f"{c_name}_{rag_index_args.meta_base}")
    corpus_index_path = os.path.join(corpus_index_dir, f"{c_name}_{rag_index_args.index_base}")

    chunks_batch = []
    metas_batch = []
    total_processed = 0
    total_filtered_korean = 0
    total_filtered_preprocessing = 0
    total_filtered_stopwords = 0  # 불용어 제거 카운트 추가
    total_filtered_duplicates = 0  # 중복 제거 카운트 추가

    # 불용어 통계를 위한 변수들
    all_removed_stopwords = []
    stopword_examples_shown = False

    # 중복 제거를 위한 세트
    seen_texts = set()

    MEMORY_BATCH_SIZE = 10000

    printi(f"🔄 Processing corpus: {c_name}")

    with open(corpus_meta_path, "w", encoding="utf-8") as meta_file:
        for ext in exts:
            txt = f"{c_name}_{c_base}.{ext}"
            path = os.path.join(c_dir, txt)
            printi(f"Loading {path}")

            if not os.path.isfile(path):
                continue

            documents = load_documents(path)
            printi(f"Loaded {len(documents)} documents from {path}")

            for title, body in tqdm(documents, desc=f"{c_name}-{ext}", unit="docs"):
                clean_body = clean(body)
                if not clean_body:
                    continue

                doc_chunks = splitter.split_text(clean_body)

                for ch in doc_chunks:
                    # 1. 한국어 텍스트 검사 (전처리 전)
                    if not is_korean_text(ch, korean_ratio_threshold):
                        total_filtered_korean += 1
                        continue

                    # 2. 가벼운 전처리 + 불용어 제거 적용
                    processed_ch, removed_stopwords = light_preprocess_text(ch)

                    # 불용어 통계 수집
                    if removed_stopwords:
                        all_removed_stopwords.extend(removed_stopwords)

                        # 처음 몇 개 예시만 출력
                        if not stopword_examples_shown and len(all_removed_stopwords) >= 10:
                            unique_stopwords = list(set(all_removed_stopwords))[:20]
                            printi(f"📋 제거된 불용어 예시 (상위 20개): {', '.join(unique_stopwords)}")
                            stopword_examples_shown = True

                    # 3. 전처리 후 빈 텍스트 체크
                    if not processed_ch or len(processed_ch.strip()) < 15:
                        total_filtered_preprocessing += 1
                        continue

                    # 4. 토큰 수 체크
                    estimated_tokens = estimate_token_count(processed_ch)
                    if estimated_tokens < MIN_TOKEN_COUNT:
                        total_filtered_preprocessing += 1
                        continue

                    # 5. 불용어 제거 후 텍스트가 너무 짧아졌는지 체크
                    if len(processed_ch.split()) < 3:  # 단어가 3개 미만이면 제외
                        total_filtered_stopwords += 1
                        continue

                    # 6. 중복 텍스트 체크 (새로 추가)
                    if processed_ch in seen_texts:
                        total_filtered_duplicates += 1
                        continue

                    # 중복이 아닌 경우 세트에 추가
                    seen_texts.add(processed_ch)

                    # 저장과 임베딩 모두 같은 전처리된 텍스트 사용
                    chunks_batch.append(processed_ch)

                    # 메타데이터에 전처리된 텍스트 저장
                    meta = Metadata(
                        corpus=c_name,
                        split=ext,
                        title=title,
                        text=processed_ch  # 전처리된 텍스트 저장
                    )

                    metas_batch.append(meta)

                    # 배치 크기에 도달하면 처리
                    if len(chunks_batch) >= MEMORY_BATCH_SIZE:
                        process_batch_and_save(
                            chunks_batch, metas_batch, model, corpus_index,
                            meta_file, rag_index_args.batch_size
                        )
                        total_processed += len(chunks_batch)
                        chunks_batch = []
                        metas_batch = []

        # 남은 배치 처리
        if chunks_batch:
            process_batch_and_save(
                chunks_batch, metas_batch, model, corpus_index,
                meta_file, rag_index_args.batch_size
            )
            total_processed += len(chunks_batch)

    # 개별 인덱스 저장
    faiss.write_index(corpus_index, corpus_index_path)

    # 불용어 통계 출력
    if all_removed_stopwords:
        from collections import Counter
        stopword_counter = Counter(all_removed_stopwords)
        most_common_stopwords = stopword_counter.most_common(15)
        printi(f"📊 가장 많이 제거된 불용어 (상위 15개):")
        for word, count in most_common_stopwords:
            printi(f"   '{word}': {count:,}회")
        printi(f"📈 총 제거된 불용어 수: {len(all_removed_stopwords):,}개")
        printi(f"📈 고유 불용어 종류: {len(set(all_removed_stopwords)):,}개")

    printi(f"✅ {c_name} 처리 완료:")
    printi(f"  - 처리된 청크: {total_processed:,}")
    printi(f"  - 한국어 필터링 제거: {total_filtered_korean:,}")
    printi(f"  - 전처리 필터링 제거: {total_filtered_preprocessing:,}")
    printi(f"  - 불용어 제거 후 제거: {total_filtered_stopwords:,}")
    printi(f"  - 중복 텍스트 제거: {total_filtered_duplicates:,}")
    printi(f"  - 고유 텍스트 수: {len(seen_texts):,}")
    printi(f"  - 인덱스 저장: {corpus_index_path}")
    printi(f"  - 메타데이터 저장: {corpus_meta_path}")

    return total_processed, total_filtered_korean, total_filtered_preprocessing + total_filtered_stopwords + total_filtered_duplicates

def merge_indices(rag_index_args: RAGIndexArgs, corpus_names: list[str]) -> None:
    """개별 인덱스들을 하나로 합치기"""
    printi("🔗 인덱스 병합 시작...")

    # 첫 번째 인덱스로 차원 확인
    first_corpus = corpus_names[0]
    first_index_path = os.path.join(rag_index_args.index_dir, first_corpus, f"{first_corpus}_{rag_index_args.index_base}")
    first_index = faiss.read_index(first_index_path)
    dim = first_index.d

    # 통합 인덱스 생성
    merged_index = faiss.IndexFlatIP(dim)
    merged_meta_path = os.path.join(rag_index_args.index_dir, rag_index_args.meta_base)
    merged_index_path = os.path.join(rag_index_args.index_dir, rag_index_args.index_base)

    total_vectors = 0

    with open(merged_meta_path, "w", encoding="utf-8") as merged_meta_file:
        for corpus_name in corpus_names:
            # 개별 인덱스 로드
            corpus_index_path = os.path.join(rag_index_args.index_dir, corpus_name, f"{corpus_name}_{rag_index_args.index_base}")
            corpus_meta_path = os.path.join(rag_index_args.index_dir, corpus_name, f"{corpus_name}_{rag_index_args.meta_base}")

            if not os.path.exists(corpus_index_path):
                printi(f"⚠️ {corpus_index_path} 파일을 찾을 수 없습니다.")
                continue

            corpus_index = faiss.read_index(corpus_index_path)

            # 벡터 복사
            vectors = np.zeros((corpus_index.ntotal, dim), dtype=np.float32)
            corpus_index.reconstruct_n(0, corpus_index.ntotal, vectors)
            merged_index.add(vectors)

            # 메타데이터 복사
            with open(corpus_meta_path, "r", encoding="utf-8") as corpus_meta_file:
                for line in corpus_meta_file:
                    merged_meta_file.write(line)

            total_vectors += corpus_index.ntotal
            printi(f"  - {corpus_name}: {corpus_index.ntotal:,} vectors 병합")

    # 통합 인덱스 저장
    faiss.write_index(merged_index, merged_index_path)

    printi(f"✅ 인덱스 병합 완료:")
    printi(f"  - 총 벡터 수: {total_vectors:,}")
    printi(f"  - 통합 인덱스: {merged_index_path}")
    printi(f"  - 통합 메타데이터: {merged_meta_path}")

def main(rag_index_args: RAGIndexArgs, process_separately: bool = True):
    """
    메인 함수 - 가벼운 전처리로 문맥 보존 + 불용어 제거 + 중복 제거
    """
    # 배치 크기 설정 (메모리에 맞게 조정)
    KOREAN_RATIO_THRESHOLD = 0.5  # 한국어 비율 임계값을 낮춤 (50%)

    os.makedirs(rag_index_args.index_dir, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_index_args.chunk_size,
        chunk_overlap=rag_index_args.chunk_overlap
    )

    # 모델 미리 로드
    printi("Loading embedding model...")
    model = SentenceTransformer(
        rag_index_args.model_name,
        device="cuda"
    )

    printi(f"🔧 전처리 설정:")
    printi(f"   - 가벼운 전처리: HTML/URL 제거, 극단적 특수문자만 제거")
    printi(f"   - 한자 제거: 모든 한자 문자 제거")
    printi(f"   - 불용어 제거: ko-ww-stopwords {'사용' if STOPWORDS_AVAILABLE else '미사용'}")
    printi(f"   - 중복 제거: 동일한 텍스트 청크 제거")
    printi(f"   - 문맥 보존: 문장부호, 조사, 어미 유지")
    printi(f"   - 한국어 비율 임계값: {KOREAN_RATIO_THRESHOLD*100}%")
    printi(f"   - 개별 처리: {'예' if process_separately else '아니오'}")

    if process_separately:
        # 개별 처리 모드
        total_processed_all = 0
        total_filtered_korean_all = 0
        total_filtered_preprocessing_all = 0
        corpus_names = []

        for corpus_info in rag_index_args.raw_text_dir:
            corpus_name = os.path.basename(corpus_info["dir"])
            corpus_names.append(corpus_name)

            processed, filtered_korean, filtered_preprocessing = process_single_corpus(
                corpus_info, rag_index_args, model, splitter, KOREAN_RATIO_THRESHOLD
            )

            total_processed_all += processed
            total_filtered_korean_all += filtered_korean
            total_filtered_preprocessing_all += filtered_preprocessing

        # 개별 인덱스들을 하나로 병합
        merge_indices(rag_index_args, corpus_names)

        total_input = total_processed_all + total_filtered_korean_all + total_filtered_preprocessing_all

        printi(f"\n📊 전체 최종 결과:")
        printi(f"✅ 총 입력 청크: {total_input:,}")
        printi(f"✅ 처리된 청크: {total_processed_all:,} ({total_processed_all/total_input*100:.1f}%)")
        printi(f"🚫 한국어 필터링 제거: {total_filtered_korean_all:,} ({total_filtered_korean_all/total_input*100:.1f}%)")
        printi(f"🚫 전처리 필터링 제거: {total_filtered_preprocessing_all:,} ({total_filtered_preprocessing_all/total_input*100:.1f}%)")

    else:
        # 기존 통합 처리 모드 (이전 코드와 동일)
        printi("⚠️ 통합 처리 모드는 메모리 사용량이 높을 수 있습니다.")

if __name__ == "__main__":
    rag_index_args = RAGIndexArgs()
    # process_separately=True로 설정하면 개별 처리 후 병합
    main(rag_index_args=rag_index_args, process_separately=True)
