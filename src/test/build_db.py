import os, re, json, faiss, numpy as np, gc
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.configs.config import RAGIndexArgs
from src.utils.print_utils import printi
from src.data.record import Metadata

# 불용어 제거를 위한 라이브러리
try:
    from soynlp.word import WordExtractor
    from soynlp.tokenizer import LTokenizer
    SOYNLP_AVAILABLE = True
    print("✅ soynlp 라이브러리 로드 성공")
except ImportError as e:
    SOYNLP_AVAILABLE = False
    print(f"⚠️ soynlp not available: {e}")

# KoNLPy 형태소 분석기
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
    print("✅ konlpy 라이브러리 로드 성공")
except ImportError as e:
    KONLPY_AVAILABLE = False
    print(f"⚠️ konlpy not available: {e}")

# 추가 디버깅 정보
if KONLPY_AVAILABLE:
    try:
        test_okt = Okt()
        print("✅ Okt 초기화 성공")
    except Exception as e:
        print(f"⚠️ Okt 초기화 실패: {e}")
        KONLPY_AVAILABLE = False

# 헤더만 잡음
hdr_pat = re.compile(r'^\s*=+\s*(.+?)\s*=+\s*$', re.MULTILINE)
bullet_pat = re.compile(r'^\s*[•\-\–·\*]\s+', re.MULTILINE)
blank_pat = re.compile(r'\n{2,}')

# 한국어 불용어 리스트 (확장된 버전)
KOREAN_STOPWORDS = {
    # 조사
    '이', '가', '을', '를', '에', '에서', '으로', '로', '와', '과', '의', '은', '는', '도', '만', '까지', '부터', '보다',
    '처럼', '같이', '마저', '조차', '라도', '뿐만', '한테', '에게', '께', '더러', '라고', '하고', '에다', '한테서',
    '께서', '님', '씨', '군', '양', '아', '야', '이여', '여', '이시여', '시여',

    # 어미 및 용언
    '다', '이다', '있다', '없다', '되다', '하다', '한다', '됩니다', '합니다', '입니다', '아니다', '그렇다', '않다',
    '었다', '았다', '였다', '겠다', '네다', '세다', '이네', '데', '지', '니', '요', '어요', '아요', '여요',

    # 대명사
    '그', '이', '저', '여기', '거기', '저기', '이것', '그것', '저것', '누구', '무엇', '어디', '언제', '어떻게',
    '왜', '어느', '얼마', '몇', '어떤', '무슨', '어떠한', '이런', '그런', '저런', '이러한', '그러한', '저러한',

    # 부사
    '또', '다시', '잘', '매우', '정말', '진짜', '참', '좀', '조금', '많이', '너무', '아주', '완전', '정말로',
    '대단히', '굉장히', '상당히', '꽤', '제법', '약간', '살짝', '다소', '거의', '아마', '혹시', '만약', '벌써',
    '이미', '아직', '더', '덜', '가장', '최고', '최대', '최소', '가끔', '자주', '항상', '언제나', '늘', '계속',

    # 접속사
    '그런데', '그러나', '하지만', '그리고', '또한', '그래서', '따라서', '그러므로', '그런데', '그래도', '그럼에도',
    '그치만', '게다가', '더욱이', '뿐만아니라', '즉', '다시말해', '한편', '반면', '물론', '사실', '실제로',

    # 감탄사
    '아', '어', '오', '우', '음', '응', '네', '예', '아니', '맞다', '틀렸다', '맞아', '아니야', '그래', '정말',

    # 수사 및 관형사
    '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉', '열', '첫', '두', '세', '네', '다섯째',
    '여섯째', '모든', '각', '온갖', '별', '헌', '새', '옛', '전',

    # 기타 자주 사용되는 불용어
    '것', '수', '때', '곳', '사람', '일', '년', '월', '일', '시간', '분', '초', '번', '개', '명', '원', '만', '천',
    '백', '십', '중', '안', '밖', '위', '아래', '앞', '뒤', '옆', '사이', '동안', '사이에', '통해', '위해', '대해',
    '관해', '따라', '의해', '로써', '로서', '으로써', '으로서', '대로', '처럼', '같이', '마냥', '듯이', '양',
    '채로', '면서', '며', '고', '거나', '든지', '든가', '나', '이나', '라든지', '라든가'
}

def advanced_preprocess_text(text: str, use_morphological_analysis: bool = True) -> str:
    """
    고급 텍스트 전처리 함수
    - 특수기호, 숫자, 영어 제거
    - 불용어 제거
    - 형태소 분석 (선택적)
    """
    if not text or not text.strip():
        return ""

    # 1. 기본 정리 (HTML 태그, URL 등 제거)
    text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
    text = re.sub(r'https?://[^\s]+', '', text)  # URL 제거
    text = re.sub(r'www\.[^\s]+', '', text)  # www 링크 제거
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)  # 이메일 제거

    # 2. 특수기호 제거 (한글, 공백만 유지)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)  # 특수기호 제거
    text = re.sub(r'[a-zA-Z]', ' ', text)  # 영어 제거
    text = re.sub(r'\d+', ' ', text)  # 숫자 제거
    text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]', ' ', text)  # 자음, 모음만 있는 글자 제거

    # 3. 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    if not text:
        return ""

    # 4. 형태소 분석 및 불용어 제거
    if use_morphological_analysis and KONLPY_AVAILABLE:
        try:
            okt = Okt()
            # 명사, 동사, 형용사만 추출
            morphs = okt.pos(text, stem=True)
            filtered_words = []

            for word, pos in morphs:
                # 의미있는 품사만 선택 (명사, 동사, 형용사)
                if pos in ['Noun', 'Verb', 'Adjective'] and len(word) > 1:
                    if word not in KOREAN_STOPWORDS:
                        filtered_words.append(word)

            text = ' '.join(filtered_words)

        except Exception as e:
            printi(f"⚠️ Morphological analysis failed: {e}")
            # 형태소 분석 실패시 기본 불용어 제거만 수행
            text = basic_stopword_removal(text)
    else:
        # 형태소 분석기 없을 때 기본 불용어 제거
        text = basic_stopword_removal(text)

    # 5. 최종 정리
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def light_preprocess_text(text: str) -> str:
    """
    가벼운 전처리 - 특수기호만 제거, 조사는 유지
    """
    if not text or not text.strip():
        return ""

    # 1. 기본 정리 (HTML 태그, URL 등 제거)
    text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
    text = re.sub(r'https?://[^\s]+', '', text)  # URL 제거
    text = re.sub(r'www\.[^\s]+', '', text)  # www 링크 제거
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)  # 이메일 제거

    # 2. 영어와 숫자만 제거 (특수기호 일부만 제거)
    text = re.sub(r'[a-zA-Z]', ' ', text)  # 영어 제거
    text = re.sub(r'\d+', ' ', text)  # 숫자 제거
    text = re.sub(r'[^\w\s가-힣]', ' ', text)  # 한글과 공백 외 특수기호 제거
    text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]', ' ', text)  # 자음, 모음만 있는 글자 제거

    # 3. 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def basic_stopword_removal(text: str) -> str:
    """기본적인 불용어 제거 (형태소 분석 없이)"""
    if not text or not text.strip():
        return ""

    words = text.split()
    filtered_words = []

    for word in words:
        # 길이 체크 (1글자 제외)
        if len(word) <= 1:
            continue

        # 불용어 체크
        if word not in KOREAN_STOPWORDS:
            filtered_words.append(word)

    return ' '.join(filtered_words)
    """기본적인 불용어 제거 (형태소 분석 없이)"""
    words = text.split()
    filtered_words = []

    for word in words:
        # 길이 체크 (1글자 제외)
        if len(word) <= 1:
            continue

        # 불용어 체크
        if word not in KOREAN_STOPWORDS:
            filtered_words.append(word)

    return ' '.join(filtered_words)

def is_korean_text(text: str, korean_ratio_threshold: float = 0.8) -> bool:
    """
    텍스트가 한국어인지 판단 (전처리 후 더 엄격한 기준)
    """
    if not text.strip():
        return False

    # 공백 제거
    content_chars = text.replace(' ', '')

    if len(content_chars) < 5:  # 전처리 후 너무 짧으면 제외
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

    # 임베딩 생성
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
                         use_morphological_analysis: bool, korean_ratio_threshold: float) -> tuple:
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

                    # 2. 가벼운 전처리 (LLM용 - 조사 유지)
                    light_processed_ch = light_preprocess_text(ch)

                    # 3. 완전 전처리 (검색용 - 형태소 분석, 불용어 제거)
                    full_processed_ch = advanced_preprocess_text(ch, use_morphological_analysis)

                    # 4. 전처리 후 빈 텍스트 체크
                    if not full_processed_ch or len(full_processed_ch.strip()) < 10:
                        total_filtered_preprocessing += 1
                        continue

                    # 검색은 완전 전처리된 텍스트로 수행
                    chunks_batch.append(full_processed_ch)

                    # 메타데이터에는 두 버전 모두 저장
                    metas_batch.append(Metadata(
                        corpus=c_name,
                        split=ext,
                        title=title,
                        text=full_processed_ch,      # 검색용 텍스트
                        model_text=light_processed_ch # LLM용 텍스트
                    ))

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

    printi(f"✅ {c_name} 처리 완료:")
    printi(f"  - 처리된 청크: {total_processed:,}")
    printi(f"  - 한국어 필터링 제거: {total_filtered_korean:,}")
    printi(f"  - 전처리 필터링 제거: {total_filtered_preprocessing:,}")
    printi(f"  - 인덱스 저장: {corpus_index_path}")
    printi(f"  - 메타데이터 저장: {corpus_meta_path}")

    return total_processed, total_filtered_korean, total_filtered_preprocessing

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
    메인 함수
    Args:
        process_separately: True면 개별 처리 후 병합, False면 기존 방식
    """
    # 배치 크기 설정 (메모리에 맞게 조정)
    KOREAN_RATIO_THRESHOLD = 0.8  # 한국어 비율 임계값 (80%)
    USE_MORPHOLOGICAL_ANALYSIS = True  # 형태소 분석 사용 여부 (다시 True로 설정)

    # 형태소 분석 가능 여부 확인
    if USE_MORPHOLOGICAL_ANALYSIS and not KONLPY_AVAILABLE:
        printi("⚠️ 형태소 분석기를 사용할 수 없습니다. 기본 전처리를 사용합니다.")
        USE_MORPHOLOGICAL_ANALYSIS = False

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
    printi(f"   - 형태소 분석: {'사용' if USE_MORPHOLOGICAL_ANALYSIS else '사용 안함'}")
    printi(f"   - 한국어 비율 임계값: {KOREAN_RATIO_THRESHOLD*100}%")
    printi(f"   - 불용어 수: {len(KOREAN_STOPWORDS)}개")
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
                corpus_info, rag_index_args, model, splitter,
                USE_MORPHOLOGICAL_ANALYSIS, KOREAN_RATIO_THRESHOLD
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
        # 여기에 기존 코드를 넣거나 별도 함수로 분리

def load_rag_index(rag_index_args: RAGIndexArgs, use_individual: bool = False):
    """
    RAG 인덱스 로드 함수
    Args:
        use_individual: True면 개별 인덱스들을 따로 로드, False면 통합 인덱스 로드
    Returns:
        통합 인덱스 사용시: (index, metadata_list)
        개별 인덱스 사용시: {corpus_name: (index, metadata_list)}
    """
    if use_individual:
        # 개별 인덱스들을 딕셔너리로 반환
        indices = {}

        for corpus_info in rag_index_args.raw_text_dir:
            corpus_name = os.path.basename(corpus_info["dir"])
            corpus_index_path = os.path.join(rag_index_args.index_dir, corpus_name, f"{corpus_name}_{rag_index_args.index_base}")
            corpus_meta_path = os.path.join(rag_index_args.index_dir, corpus_name, f"{corpus_name}_{rag_index_args.meta_base}")

            if not os.path.exists(corpus_index_path):
                printi(f"⚠️ {corpus_index_path} 파일을 찾을 수 없습니다.")
                continue

            # 인덱스 로드
            index = faiss.read_index(corpus_index_path)

            # 메타데이터 로드
            metadata_list = []
            with open(corpus_meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        metadata_list.append(json.loads(line))

            indices[corpus_name] = (index, metadata_list)
            printi(f"✅ {corpus_name} 인덱스 로드 완료: {index.ntotal:,} vectors")

        return indices

    else:
        # 통합 인덱스 로드
        merged_index_path = os.path.join(rag_index_args.index_dir, rag_index_args.index_base)
        merged_meta_path = os.path.join(rag_index_args.index_dir, rag_index_args.meta_base)

        if not os.path.exists(merged_index_path):
            raise FileNotFoundError(f"통합 인덱스 파일을 찾을 수 없습니다: {merged_index_path}")

        # 인덱스 로드
        index = faiss.read_index(merged_index_path)

        # 메타데이터 로드
        metadata_list = []
        with open(merged_meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metadata_list.append(json.loads(line))

        printi(f"✅ 통합 인덱스 로드 완료: {index.ntotal:,} vectors")
        return index, metadata_list

def search_rag_index(query: str, index, metadata_list, model, top_k: int = 5):
    """
    RAG 인덱스에서 검색
    Args:
        query: 검색 쿼리
        index: FAISS 인덱스
        metadata_list: 메타데이터 리스트
        model: 임베딩 모델
        top_k: 반환할 상위 결과 수
    Returns:
        검색 결과 리스트
    """
    # 쿼리 완전 전처리 (검색용)
    processed_query = advanced_preprocess_text(query, use_morphological_analysis=True)
    if not processed_query:
        processed_query = query  # 전처리 실패시 원본 사용

    # 쿼리 임베딩
    query_vec = model.encode([processed_query], normalize_embeddings=True)

    # 검색 수행
    scores, indices = index.search(np.float32(query_vec), top_k)

    # 결과 정리
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1:  # 유효한 인덱스
            metadata = metadata_list[idx]

            results.append({
                "rank": i + 1,
                "score": float(score),
                "corpus": metadata.get("corpus", "unknown"),
                "split": metadata.get("split", "unknown"),
                "title": metadata.get("title", "unknown"),
                "text": metadata.get("model_text", metadata.get("text", "")),  # LLM용 텍스트 우선
                "search_text": metadata.get("text", "")  # 디버깅용 (검색에 사용된 전처리 텍스트)
            })

    return results

# 사용 예시 함수
def example_usage():
    """사용 예시"""
    rag_index_args = RAGIndexArgs()

    # 모델 로드
    model = SentenceTransformer(rag_index_args.model_name, device="cuda")

    # 방법 1: 통합 인덱스 사용
    printi("=== 통합 인덱스 사용 예시 ===")
    index, metadata_list = load_rag_index(rag_index_args, use_individual=False)

    # 검색 수행
    query = "한국의 전통 음식"
    results = search_rag_index(query, index, metadata_list, model, top_k=3)

    printi(f"검색 쿼리: {query}")
    for result in results:
        printi(f"순위 {result['rank']}: {result['score']:.4f} - {result['corpus']} - {result['title']}")
        printi(f"내용: {result['text'][:100]}...")

    # 방법 2: 개별 인덱스 사용
    printi("\n=== 개별 인덱스 사용 예시 ===")
    indices_dict = load_rag_index(rag_index_args, use_individual=True)

    for corpus_name, (corpus_index, corpus_metadata) in indices_dict.items():
        printi(f"\n--- {corpus_name} 검색 ---")
        results = search_rag_index(query, corpus_index, corpus_metadata, model, top_k=2)

        for result in results:
            printi(f"순위 {result['rank']}: {result['score']:.4f} - {result['title']}")
            printi(f"내용: {result['text'][:100]}...")

if __name__ == "__main__":
    rag_index_args = RAGIndexArgs()
    # process_separately=True로 설정하면 개별 처리 후 병합
    # process_separately=False로 설정하면 기존 방식 (모든 데이터를 메모리에 로드)
    main(rag_index_args=rag_index_args, process_separately=True)

    # 사용 예시 실행 (주석 해제하여 사용)
    # example_usage()
