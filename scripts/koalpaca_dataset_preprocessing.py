import os
import json
from sklearn.model_selection import train_test_split

# 원본 데이터 디렉터리
data_dir = "datasets/KoAlpaca-v1.1a"
# 변환 후 저장할 디렉터리
target_data_dir = "datasets/KoAlpaca-v1.1a_converted"

current_id = -1

def convert_to_dataset_format(item):
    global current_id
    current_id += 1

    """주어진 item을 새로운 데이터셋 포맷으로 변환합니다."""
    question = item.get("instruction", "")
    answer = item.get("output", "")

    return {
        "id": f"{current_id}",
        "input": {
            "category": "",
            "domain": "",
            "question_type": "서술형",
            "topic_keyword": "",
            "question": question,
        },
        "output": {
            "answer": answer
        }
    }

if __name__ == "__main__":
    # 원본 데이터 로드
    with open(f"{data_dir}/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # 저장할 디렉터리 생성
    os.makedirs(target_data_dir, exist_ok=True)

    # 데이터를 9:1 비율로 분할 (train: 90%, validation: 10%)
    train, val = train_test_split(train_data, test_size=0.1, random_state=42)

    # train, val 데이터를 새로운 포맷으로 변환
    converted_train = [convert_to_dataset_format(item) for item in train]
    converted_val = [convert_to_dataset_format(item) for item in val]

    # 변환된 train 데이터를 train.json으로 저장
    with open(f"{target_data_dir}/train.json", "w", encoding="utf-8") as f:
        json.dump(converted_train, f, ensure_ascii=False, indent=4)

    # 변환된 validation 데이터를 dev.json으로 저장
    with open(f"{target_data_dir}/dev.json", "w", encoding="utf-8") as f:
        json.dump(converted_val, f, ensure_ascii=False, indent=4)

    print("데이터 변환 및 저장이 완료되었습니다. ✅")
    print(f"Train data ({len(converted_train)}개): {os.path.join(target_data_dir, 'train.json')}")
    print(f"Dev data ({len(converted_val)}개): {os.path.join(target_data_dir, 'dev.json')}")
