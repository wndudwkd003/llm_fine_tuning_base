import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

csv_path = "datasets/1_corpus_download_01.엑소브레인 QA Datasets/ETRI QA Datasets/1.#Ud034#Uc988QA Datasets/주관식.csv"
save_dir = "datasets/etri_qa_c_converted"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)

converted = []
for idx, row in df.iterrows():
    question = str(row["질문(Q_string)"]).strip()
    answer = str(row["정답(A_ans)"]).strip()

    converted.append({
        "id": f"주관식_{idx+1}",
        "input": {
            "category": "",
            "domain": "",
            "question_type": "단답형",
            "topic_keyword": "",
            "question": question
        },
        "output": {
            "answer": answer
        }
    })

# 9:1 분할 저장
train_data, dev_data = train_test_split(converted, test_size=0.1, random_state=42)

with open(os.path.join(save_dir, "train.json"), "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(os.path.join(save_dir, "dev.json"), "w", encoding="utf-8") as f:
    json.dump(dev_data, f, ensure_ascii=False, indent=2)

print(f"train.json: {len(train_data)}개, dev.json: {len(dev_data)}개 저장 완료.")
