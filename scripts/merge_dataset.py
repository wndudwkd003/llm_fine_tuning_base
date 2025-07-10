import os, json, shutil, random

def load_json(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def save_json(data, p):
    with open(p, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

SHUFFLE = True  # True면 각 병합 리스트를 랜덤 셔플

if __name__ == "__main__":
    original_dataset = "datasets/sub_3_data_korean_culture_qa_V1.0_preprocessed"
    other_datasets = [
        "datasets/KoAlpaca-v1.1a_converted",
        # "datasets/CLIcK_converted",
        # "datasets/squad_kor_v1_converted",
        # "datasets/tydiqa-goldp_converted",
        # "datasets/KorWikiTQ_ko_converted",
        # "datasets/KMMLU_converted",
        # "datasets/etri_mrc_converted",
        # "datasets/etri_qa_a_converted",
        # "datasets/etri_qa_b_converted",
        # "datasets/etri_qa_c_converted",
    ]

    target_dataset = "datasets/merged_dataset_no_aug_v1"
    os.makedirs(target_dataset, exist_ok=True)

    # ① test.json은 원본 그대로 복사
    shutil.copy2(os.path.join(original_dataset, "test.json"),
                 os.path.join(target_dataset, "test.json"))
    print(f"Copied test.json from {original_dataset} to {target_dataset}")

    # ② train.json, dev.json 등 나머지 개별 파일 병합
    for fname in ["train.json", "dev.json"]:
        merged, next_id = [], 0
        source_dirs = [original_dataset] + other_datasets
        for d in source_dirs:
            src_path = os.path.join(d, fname)
            if os.path.exists(src_path):
                for item in load_json(src_path):
                    item["id"] = str(next_id)
                    next_id += 1
                    merged.append(item)
        if SHUFFLE:
            random.shuffle(merged)
        save_json(merged, os.path.join(target_dataset, fname))
        print(f"Merged {fname} from {len(source_dirs)} sources into {target_dataset} ({len(merged)}개)")
