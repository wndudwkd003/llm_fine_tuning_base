import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# 객관식

csv_path = "datasets/1_corpus_download_01.엑소브레인 QA Datasets/ETRI QA Datasets/1.#Ud034#Uc988QA Datasets/객관식.csv"
save_dir = "datasets/etri_qa_a_converted"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)

converted = []
for idx, row in df.iterrows():
    question = str(row["질문(Q_string)"]).strip()
    choices = [
        str(row["보기()E_string"]).strip(),
        str(row["보기()E_string.1"]).strip(),
        str(row["보기()E_string.2"]).strip(),
        str(row["보기()E_string.3"]).strip()
    ]
    answer_text = str(row["정답(A_ans)"]).strip()
    try:
        answer_index = choices.index(answer_text) + 1
    except ValueError:
        continue  # 정답이 보기 중에 없으면 제외

    choice_str = " ".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
    full_question = f"{question} {choice_str}"

    converted.append({
        "id": f"객관식_{idx+1}",
        "input": {
            "category": "전통 문화 및 일반상식",  # 고정값 또는 나중에 분류
            "domain": str(row.get("위키피디아 제목(AP_title)", "")).strip(),
            "question_type": "선다형",
            "topic_keyword": "",  # 필요시 키워드 추출 추가
            "question": full_question
        },
        "output": {
            "answer": str(answer_index)
        }
    })

# 전체 데이터를 train으로 저장 (분할하지 않음)
with open(os.path.join(save_dir, "train.json"), "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

# 빈 dev.json 파일 생성
with open(os.path.join(save_dir, "dev.json"), "w", encoding="utf-8") as f:
    json.dump([], f, ensure_ascii=False, indent=2)

print(f"train.json: {len(converted)}개, dev.json: 0개 저장 완료.")
