import os, json, shutil, random

def load_json(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def save_json(data, p):
    with open(p, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

SHUFFLE = True  # True면 데이터들끼리 랜덤 셔플

if __name__ == "__main__":
    # etri_qa_a, b, c 데이터셋만 병합
    datasets = [
        "datasets/etri_qa_a_converted",
        "datasets/etri_qa_b_converted",
        "datasets/etri_qa_c_converted",
    ]

    target_dataset = "datasets/etri_qa_abc"
    os.makedirs(target_dataset, exist_ok=True)

    # train.json, dev.json, test.json 파일들을 병합
    for fname in ["train.json", "dev.json"]:
        merged, next_id = [], 0
        total_count = 0

        print(f"\n=== Processing {fname} ===")

        # 각 데이터셋에서 데이터 수집
        all_data = []
        print(f"📁 데이터 수집:")
        for d in datasets:
            src_path = os.path.join(d, fname)
            if os.path.exists(src_path):
                dataset_items = load_json(src_path)
                dataset_count = len(dataset_items)
                total_count += dataset_count
                for item in dataset_items:
                    item["id"] = str(next_id)
                    next_id += 1
                    all_data.append(item)
                print(f"   - {os.path.basename(d)}: {dataset_count:,}개")
            else:
                print(f"   - {os.path.basename(d)}: 0개 (파일 없음)")

        # 데이터 셔플
        if SHUFFLE and all_data:
            random.shuffle(all_data)
            print(f"🔀 데이터 셔플 완료: {total_count:,}개")

        # 병합된 데이터 저장
        merged = all_data
        save_json(merged, os.path.join(target_dataset, fname))

        print(f"💾 최종 병합 결과:")
        print(f"   - 전체 샘플: {len(merged):,}개")
        print(f"   - 저장 경로: {target_dataset}/{fname}")
        print("-" * 50)
