import os, json, shutil, random   # random 추가

def load_json(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def save_json(data, p):
    with open(p, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

SHUFFLE = True  # True면 각 병합 리스트를 랜덤 셔플, False면 원래 순서를 유지

if __name__ == "__main__":
    original_dataset = "datasets/sub_3_data_korean_culture_qa_V1.0_preprocessed"
    other_datasets = [
        "datasets/KoAlpaca-v1.1a_converted",
        "datasets/CLIcK_converted",
        "datasets/squad_kor_v1.2_converted",
    ]
    target_dataset = "datasets/merged_dataset_no_aug_v1"
    os.makedirs(target_dataset, exist_ok=True)

    # 폴더별 JSON 파일 목록 수집
    file_groups = {}
    for d in [original_dataset] + other_datasets:
        for fname in os.listdir(d):
            if fname.endswith(".json"):
                if fname == "test.json" and d != original_dataset:  # test.json은 원본만 사용
                    continue
                file_groups.setdefault(fname, []).append(os.path.join(d, fname))

    # ① test.json: 그대로 복사
    shutil.copy2(os.path.join(original_dataset, "test.json"),
                 os.path.join(target_dataset, "test.json"))
    print(f"Copied test.json from {original_dataset} to {target_dataset}")

    # ② train.json, dev.json 등 나머지 파일 병합
    for fname, paths in file_groups.items():
        if fname == "test.json":  # 이미 처리
            continue
        merged, next_id = [], 0
        for p in paths:
            for item in load_json(p):
                item["id"] = str(next_id)
                next_id += 1
                merged.append(item)
        if SHUFFLE:  # 셔플 옵션 적용
            random.shuffle(merged)
        save_json(merged, os.path.join(target_dataset, fname))
        print(f"Merged {fname} from {len(paths)} sources into {target_dataset}")
