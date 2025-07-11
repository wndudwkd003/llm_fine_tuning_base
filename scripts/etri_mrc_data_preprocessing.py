import os, json
from sklearn.model_selection import train_test_split

def convert_to_dataset_format(context, title, qas):
    """하나의 질문-답변 쌍을 LLM 학습용 포맷으로 변환"""
    results = []
    for qa in qas:
        question = qa["question"]
        answer_text = qa["answers"][0]["text"] if qa["answers"] else ""
        full_question = f"{question} [배경] {context}"
        results.append({
            "id": qa["id"],
            "input": {
                "category": "",
                "domain": title,
                "question_type": "단답형",
                "topic_keyword": "",
                "question": full_question
            },
            "output": {
                "answer": answer_text
            }
        })
    return results

if __name__ == "__main__":
    data_dir = "datasets/etri_mrc"
    target_dir = data_dir + "_converted"
    os.makedirs(target_dir, exist_ok=True)

    # 데이터 로딩
    with open(os.path.join(data_dir, "train.json"), encoding="utf-8") as f:
        raw = json.load(f)

    merged_data = []
    for item in raw["data"]:
        title = item["title"]
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]
            qas = paragraph["qas"]
            merged_data.extend(convert_to_dataset_format(context, title, qas))

    # 전체 데이터를 변환 (분할하지 않음)
    with open(os.path.join(target_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"train.json 저장 완료 ({len(merged_data)}개)")

    # 빈 dev.json 파일 생성
    with open(os.path.join(target_dir, "dev.json"), "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
        print(f"dev.json 저장 완료 (0개)")
