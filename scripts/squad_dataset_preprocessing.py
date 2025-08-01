import os, json
from sklearn.model_selection import train_test_split

def convert_to_dataset_format(item):
    """주어진 item을 새로운 데이터셋 포맷으로 변환합니다."""
    context = item.get("context", "")
    question = item.get("question", "")
    answer_text = item.get("answers", {}).get("text", [""])[0]
    new_question = f"{question} [배경] {context}"

    return {
        "id": item.get("id", ""),
        "input": {
            "category": "",
            "domain": "",
            "question_type": "단답형",
            "topic_keyword": "",
            "question": new_question,
        },
        "output": {
            "answer": answer_text
        }
    }

if __name__ == "__main__":
    data_dir = "datasets/squad_kor_v1"
    target_dir = "datasets/squad_kor_v1_converted"
    os.makedirs(target_dir, exist_ok=True)

    # train.json과 validation.json 병합
    merged_data = []
    for file_name in ["train.json", "validation.json"]:
        with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as f:
            data = json.load(f)
            merged_data.extend(data)

    # 전체 데이터를 변환 (분할하지 않음)
    converted_train = [convert_to_dataset_format(item) for item in merged_data]

    # 저장
    with open(os.path.join(target_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(converted_train, f, ensure_ascii=False, indent=4)
        print(f"Saved train.json to {target_dir} ({len(converted_train)}개)")

    # 빈 dev.json 파일 생성
    with open(os.path.join(target_dir, "dev.json"), "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)
        print(f"Saved dev.json to {target_dir} (0개)")
