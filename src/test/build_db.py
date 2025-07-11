import os, re, json, faiss, numpy as np, gc
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.configs.config import RAGIndexArgs
from src.utils.print_utils import printi
from src.data.record import Metadata

# 헤더만 잡음
hdr_pat = re.compile(r'^\s*=+\s*(.+?)\s*=+\s*$', re.MULTILINE)
bullet_pat = re.compile(r'^\s*[•\-\–·\*]\s+', re.MULTILINE)
blank_pat = re.compile(r'\n{2,}')

def is_korean_text(text: str, korean_ratio_threshold: float = 0.7) -> bool:
    """
    텍스트가 한국어인지 판단
    Args:
        text: 검사할 텍스트
        korean_ratio_threshold: 한국어 문자 비율 임계값 (기본 70%)
    Returns:
        bool: 한국어 텍스트 여부
    """
    if not text.strip():
        return False

    # 공백과 특수문자를 제외한 실제 문자들만 추출
    content_chars = re.sub(r'[\s\n\r\t\[\](){}.,!?;:"""''—\-=+*/_|\\<>&%@#$^~`]', '', text)

    if len(content_chars) < 10:  # 너무 짧은 텍스트는 건너뜀
        return True

    # 한국어 문자 (한글 + 한자) 개수 계산
    korean_chars = len(re.findall(r'[가-힣一-龯]', content_chars))

    # 한국어 비율 계산
    korean_ratio = korean_chars / len(content_chars)

    return korean_ratio >= korean_ratio_threshold

def clean(txt: str) -> str:
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

def main(rag_index_args: RAGIndexArgs):
    # 배치 크기 설정 (메모리에 맞게 조정)
    MEMORY_BATCH_SIZE = 1000  # 한 번에 처리할 청크 수
    KOREAN_RATIO_THRESHOLD = 0.7  # 한국어 비율 임계값 (70%)

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

    # 첫 번째 배치로 차원 확인을 위한 더미 임베딩
    dummy_vec = model.encode(["dummy"], normalize_embeddings=True)
    dim = dummy_vec.shape[1]

    # FAISS 인덱스 초기화
    index = faiss.IndexFlatIP(dim)

    # 메타데이터 파일 열기
    meta_path = os.path.join(rag_index_args.index_dir, rag_index_args.meta_base)

    chunks_batch = []
    metas_batch = []
    total_processed = 0
    total_filtered = 0  # 한국어 필터링으로 제거된 청크 수

    with open(meta_path, "w", encoding="utf-8") as meta_file:
        for corpus in rag_index_args.raw_text_dir:
            c_dir = corpus["dir"]
            c_name = os.path.basename(c_dir)
            c_base = corpus["base"]
            exts = corpus["ext"]

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
                        # 한국어 텍스트 검사
                        if not is_korean_text(ch, KOREAN_RATIO_THRESHOLD):
                            total_filtered += 1
                            continue  # 한국어 비율이 낮으면 제외

                        chunks_batch.append(ch)
                        metas_batch.append(Metadata(
                            corpus=c_name,
                            split=ext,
                            title=title,
                            text=ch
                        ))

                        # 배치 크기에 도달하면 처리
                        if len(chunks_batch) >= MEMORY_BATCH_SIZE:
                            process_batch_and_save(
                                chunks_batch, metas_batch, model, index,
                                meta_file, rag_index_args.batch_size
                            )
                            total_processed += len(chunks_batch)
                            chunks_batch = []
                            metas_batch = []

                printi(f"Processed {c_name}-{ext}, total so far: {total_processed}, filtered: {total_filtered}")

        # 남은 배치 처리
        if chunks_batch:
            process_batch_and_save(
                chunks_batch, metas_batch, model, index,
                meta_file, rag_index_args.batch_size
            )
            total_processed += len(chunks_batch)

    # 최종 인덱스 저장
    idx_path = os.path.join(rag_index_args.index_dir, rag_index_args.index_base)
    faiss.write_index(index, idx_path)

    printi(f"✅ Total processed: {total_processed:,} chunks")
    printi(f"🚫 Total filtered out: {total_filtered:,} chunks ({total_filtered/(total_processed+total_filtered)*100:.1f}%)")
    printi(f"✅ Index saved: {idx_path}")
    printi(f"✅ Metadata saved: {meta_path}")
    printi(f"✅ Index size: {index.ntotal:,} vectors")

if __name__ == "__main__":
    rag_index_args = RAGIndexArgs()
    main(rag_index_args=rag_index_args)
