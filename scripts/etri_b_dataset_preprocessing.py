import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

csv_path = "datasets/1_corpus_download_01.엑소브레인 QA Datasets/ETRI QA Datasets/1.#Ud034#Uc988QA Datasets/연상형.csv"
save_dir = "datasets/etri_qa_b_converted"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)

converted = []
for idx, row in df.iterrows():
    question = str(row["질문(Q_string)"]).strip()
    answer_text = str(row["정답(A_ans)"]).strip()
    question_type = str(row.get("문제유형(Q_form)", "")).strip()

    converted.append({
        "id": f"연상형_{idx+1}",
        "input": {
            "category": "",
            "domain": "",
            "question_type": "단답형",
            "topic_keyword": "",
            "question": question
        },
        "output": {
            "answer": answer_text
        }
    })

# 전체 데이터를 train으로 저장 (분할하지 않음)
with open(os.path.join(save_dir, "train.json"), "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

# 빈 dev.json 파일 생성
with open(os.path.join(save_dir, "dev.json"), "w", encoding="utf-8") as f:
    json.dump([], f, ensure_ascii=False, indent=2)

print(f"train.json: {len(converted)}개, dev.json: 0개 저장 완료.")
