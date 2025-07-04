from datasets import load_dataset
import os, json

DATASET_ID = "KorQuAD/squad_kor_v2"
OUT_DIR = "datasets/squad_kor_v2"

os.makedirs(OUT_DIR, exist_ok=True)

# trust_remote_code=True 추가
dataset_dict = load_dataset(DATASET_ID, trust_remote_code=True)
print("Available splits:", list(dataset_dict.keys()))

for split_name, ds in dataset_dict.items():
    path = os.path.join(OUT_DIR, f"{split_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([item for item in ds], f, ensure_ascii=False, indent=2)
    print(f"Saved {split_name} ({len(ds):,} rows) → {path}")
