import os, re, json, faiss, numpy as np
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


def main(rag_index_args: RAGIndexArgs):
    # max_chunks = 100

    os.makedirs(rag_index_args.index_dir, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_index_args.chunk_size,
        chunk_overlap=rag_index_args.chunk_overlap
    )

    chunks = []
    metas = []

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
            for title, body in tqdm(documents, desc=f"{c_name}-{ext}", unit="docs"):
                clean_body = clean(body)
                if not clean_body:
                    continue

                for ch in splitter.split_text(clean_body):
                    chunks.append(ch)
                    metas.append(Metadata(
                        corpus=c_name,
                        split=ext,
                        title=title,
                        text=ch
                    ))
        #             if len(chunks) >= max_chunks:
        #                 break
        #         if len(chunks) >= max_chunks:
        #             break
        #     if len(chunks) >= max_chunks:
        #         break
        # if len(chunks) >= max_chunks:
        #     break

        printi(f"Loaded {len(chunks)} records and {len(metas)} metas from {c_name} corpus\n")


    # embedding
    printi(f"Embedding {len(chunks):,} chunks…")
    model = SentenceTransformer(
        rag_index_args.model_name,
        device="cuda"
    )

    vecs = model.encode(
        chunks,
        batch_size=rag_index_args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.float32(vecs))

    idx_path = os.path.join(rag_index_args.index_dir, rag_index_args.index_base)
    meta_path = os.path.join(rag_index_args.index_dir, rag_index_args.meta_base)

    faiss.write_index(index, idx_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m.__dict__, ensure_ascii=False) + "\n")

    printi(f"saved index: {idx_path}")
    printi(f"saved meta : {meta_path}")

if __name__ == "__main__":
    rag_index_args = RAGIndexArgs()
    main(rag_index_args=rag_index_args)
