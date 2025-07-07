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
doc_hdr_pat = re.compile(r'^\s*=+\s*(.+?)\s*=+')

def clean(txt: str) -> str:
    txt = hdr_pat.sub(r'[SECTION]\1', txt)
    txt = bullet_pat.sub('', txt)
    txt = blank_pat.sub('\n', txt)
    return txt.strip()

def main(rag_index_args: RAGIndexArgs):
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

            with open(path, encoding="utf-8") as f:
                for line in tqdm(f, desc=f"{c_name}-{ext}", unit="docs"):
                    if not line.strip():
                        continue

                    m = doc_hdr_pat.match(line)
                    if not m:
                        continue

                    title = m.group(1).strip()
                    body = clean(line[m.end():].lstrip())

                    if not body:
                        continue

                    for ch in splitter.split_text(body):
                        chunks.append(ch)
                        metas.append(Metadata(
                            corpus=c_name,
                            split=ext,
                            title=title,
                            text=ch,
                        ))

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
