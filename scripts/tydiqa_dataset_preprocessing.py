import os, json
from sklearn.model_selection import train_test_split

def convert_to_dataset_format(item):
    context = item.get("passage_text", "")
    question = item.get("question_text", "")
    answer_text = item.get("answers", {}).get("text", [""])[0]
    document_title = item.get("document_title", "")

    new_question = f"{question} [배경] {context}"

    return {
        "id": item.get("id", ""),
        "input": {
            "category": "",
            "domain": document_title,
            "question_type": "단답형",
            "topic_keyword": "",
            "question": new_question,
        },
        "output": {
            "answer": answer_text
        }
    }

if __name__ == "__main__":
    data_dir = "datasets/tydiqa-goldp"
    target_dir = data_dir + "_converted"
    os.makedirs(target_dir, exist_ok=True)

    # train.json과 validation.json 병합
    merged_data = []
    for file_name in ["train.json", "dev.json"]:
        with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as f:
            data = json.load(f)
            merged_data.extend(data)

    # 9:1 분할
    train_data, dev_data = train_test_split(merged_data, test_size=0.1, random_state=42)

    # 변환
    converted_train = [convert_to_dataset_format(item) for item in train_data]
    converted_dev = [convert_to_dataset_format(item) for item in dev_data]

    # 저장
    with open(os.path.join(target_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(converted_train, f, ensure_ascii=False, indent=4)
        print(f"Saved train.json to {target_dir} ({len(converted_train)}개)")

    with open(os.path.join(target_dir, "dev.json"), "w", encoding="utf-8") as f:
        json.dump(converted_dev, f, ensure_ascii=False, indent=4)
        print(f"Saved dev.json to {target_dir} ({len(converted_dev)}개)")
