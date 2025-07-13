import os, json, shutil, random

def load_json(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def save_json(data, p):
    with open(p, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

SHUFFLE = True  # True면 other 데이터들끼리만 랜덤 셔플


"""
1:
"datasets/CLIcK_converted",


2:
"datasets/KoAlpaca-v1.1a_converted",


3:
"datasets/etri_qa_a_converted",
"datasets/etri_qa_b_converted",
"datasets/etri_qa_c_converted",


4:
"datasets/etri_mrc_converted",


5:
"datasets/KorWikiTQ_ko_converted",

6:
"datasets/KMMLU_converted",


7:
"datasets/tydiqa-goldp_converted",


8:
"datasets/squad_kor_v1_converted",

"""

"""
1-3-cot:
datasets/CLIcK_converted_cot_refined_4.1_converted


"""



if __name__ == "__main__":
    data_number = "1-3-cot"
    original_dataset = "datasets/sub_3_data_korean_culture_qa_V1.0_preprocessed_cot_refined_4.1_converted"
    other_datasets = [
        "datasets/CLIcK_converted_cot_refined_4.1_converted"
        # "datasets/CLIcK_converted",
        # "datasets/KoAlpaca-v1.1a_converted",
        # "datasets/etri_qa_a_converted",
        # "datasets/etri_qa_b_converted",
        # "datasets/etri_qa_c_converted",
        # "datasets/etri_mrc_converted",
        # "datasets/KorWikiTQ_ko_converted",
        # "datasets/KMMLU_converted",
        # "datasets/tydiqa-goldp_converted",
        # "datasets/squad_kor_v1_converted",

    ]

    target_dataset = f"datasets/merged_dataset_no_aug_v{data_number}"
    os.makedirs(target_dataset, exist_ok=True)

    # ① test.json은 원본 그대로 복사
    shutil.copy2(os.path.join(original_dataset, "test.json"),
                 os.path.join(target_dataset, "test.json"))
    print(f"Copied test.json from {original_dataset} to {target_dataset}")

    # ② train.json, dev.json 등 나머지 개별 파일 병합
    for fname in ["train.json", "dev.json"]:
        merged, next_id = [], 0
        original_count = 0

        print(f"\n=== Processing {fname} ===")

        # 1) 원본 데이터 먼저 추가
        original_path = os.path.join(original_dataset, fname)
        if os.path.exists(original_path):
            original_data = load_json(original_path)
            original_count = len(original_data)
            for item in original_data:
                item["id"] = str(next_id)
                next_id += 1
                merged.append(item)
            print(f"📁 원본 데이터: {original_count:,}개 샘플")
        else:
            print(f"📁 원본 데이터: 0개 샘플 (파일 없음)")

        # 2) other 데이터들 수집 후 섞기
        other_data = []
        total_other_count = 0
        print(f"📁 추가 데이터:")
        for d in other_datasets:
            src_path = os.path.join(d, fname)
            if os.path.exists(src_path):
                dataset_items = load_json(src_path)
                dataset_count = len(dataset_items)
                total_other_count += dataset_count
                for item in dataset_items:
                    item["id"] = str(next_id)
                    next_id += 1
                    other_data.append(item)
                print(f"   - {os.path.basename(d)}: {dataset_count:,}개")
            else:
                print(f"   - {os.path.basename(d)}: 0개 (파일 없음)")

        # 3) other 데이터들끼리만 셔플
        if SHUFFLE and other_data:
            random.shuffle(other_data)
            print(f"🔀 추가 데이터 셔플 완료: {total_other_count:,}개")

        # 4) 원본 + other(셔플된) 순서로 병합
        merged.extend(other_data)

        save_json(merged, os.path.join(target_dataset, fname))

        print(f"💾 최종 병합 결과:")
        print(f"   - 원본 데이터: {original_count:,}개")
        print(f"   - 추가 데이터: {total_other_count:,}개")
        print(f"   - 전체 샘플: {len(merged):,}개")
        print(f"   - 저장 경로: {target_dataset}/{fname}")
        print("-" * 50)
