import os
import json
from sklearn.model_selection import train_test_split

# 원본 데이터 디렉터리
data_dir = "datasets/CLIcK"
# 변환 후 저장할 디렉터리
target_data_dir = "datasets/CLIcK_converted"

def convert_to_dataset_format(item):
    """주어진 item을 새로운 데이터셋 포맷으로 변환합니다."""
    paragraph = item.get("paragraph", "")
    question = item.get("question", "")
    choices = item.get("choices", [])
    answer = item.get("answer", "")

    new_question = f"{paragraph + ' ' if paragraph else ''}{question}".strip()

    add_question = []
    for index, choice in enumerate(choices):
        add_question.append(f"{index+1}. {choice}")
    add_question = " ".join(add_question)

    new_question = f"{new_question} 알맞은 것을 골라보세요? {add_question}".strip()
    # 정답을 선택지의 인덱스+1 (1-based)로 변환
    new_answer = f"{choices.index(answer) + 1}"

    return {
        "id": item.get("id", ""),
        "input": {
            "category": "",
            "domain": "",
            "question_type": "선다형",
            "topic_keyword": "",
            "question": new_question,
        },
        "output": {
            "answer": new_answer
        }
    }

if __name__ == "__main__":
    # 원본 데이터 로드
    with open(f"{data_dir}/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # 저장할 디렉터리 생성
    os.makedirs(target_data_dir, exist_ok=True)

    # 전체 데이터를 새로운 포맷으로 변환 (분할하지 않음)
    converted_train = [convert_to_dataset_format(item) for item in train_data]

    # 변환된 전체 데이터를 train.json으로 저장
    with open(f"{target_data_dir}/train.json", "w", encoding="utf-8") as f:
        json.dump(converted_train, f, ensure_ascii=False, indent=4)

    # 빈 dev.json 파일 생성
    with open(f"{target_data_dir}/dev.json", "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

    print("데이터 변환 및 저장이 완료되었습니다. ✅")
    print(f"Train data ({len(converted_train)}개): {os.path.join(target_data_dir, 'train.json')}")
    print(f"Dev data (0개): {os.path.join(target_data_dir, 'dev.json')}")
