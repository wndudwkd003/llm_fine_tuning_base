from datasets import load_dataset, get_dataset_config_names
import os
import json

DATASET_ID = "khalidalt/tydiqa-goldp"
OUT_DIR = "datasets/tydiqa-goldp"
os.makedirs(OUT_DIR, exist_ok=True)

# 이 데이터셋에서 사용 가능한 모든 config 이름 목록을 가져옵니다.
available_configs = get_dataset_config_names(DATASET_ID, trust_remote_code=True)
print(f"'{DATASET_ID}'에서 사용 가능한 Config: {available_configs}")

# 각 config를 순회하며 다운로드합니다.
for config_name in available_configs:
    try:
        print(f"\n--- '{config_name}' 다운로드 시작 ---")
        dataset_dict = load_dataset(DATASET_ID, config_name, trust_remote_code=True)

        for split_name, ds in dataset_dict.items():
            # 저장 파일명에 config 이름을 포함시켜 구분합니다.
            path = os.path.join(OUT_DIR, f"{config_name}_{split_name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([item for item in ds], f, ensure_ascii=False, indent=2)
            print(f"Saved {split_name} ({len(ds):,} rows) → {path}")

    except Exception as e:
        print(f"'{config_name}' 처리 중 오류 발생: {e}")

print("\n모든 작업이 완료되었습니다. ✅")
