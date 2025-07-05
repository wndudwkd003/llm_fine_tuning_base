# v1


import os, json


def convert_to_dataset_format(item):
    """주어진 item을 새로운 데이터셋 포맷으로 변환합니다."""
    context = item.get("context", "")
    question = item.get("question", "")
    answer_text = item.get("answers", {}).get("text", [""])[0]
    new_question = f"""{question}"""

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


remap_keys = {
    "validation.json": "dev.json"
}

if __name__ == "__main__":
    data_dir = "datasets/squad_kor_v1"
    target_dir = "datasets/squad_kor_v1.1_converted"

    os.makedirs(target_dir, exist_ok=True)

    target_file = ["train.json", "validation.json"]

    for file_name in target_file:
        with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as f:
            data = json.load(f)

        converted_data = [convert_to_dataset_format(item) for item in data]

        with open(os.path.join(target_dir, (file_name if file_name not in remap_keys.keys() else remap_keys.get(file_name, file_name))), "w", encoding="utf-8") as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=4)
            print(f"Converted {file_name} and saved to {target_dir}")


