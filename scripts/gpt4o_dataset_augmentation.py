import os
import json
import openai


def main():
    dataset_dir = "datasets/sub_3_data_korean_culture_qa_V1.0"
    target_data = "train.json"
    path = os.path.join(dataset_dir, target_data)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        question = item["input"]["question"]
        answer = item["output"]["answer"]
        prompt = f"질문: {question}\n정답: {answer}\n 이러한 질의응답에 대하여 왜 그런지 이유를 한국어로 자세히 설명해 주세요. 단, 이유만 설명하고 추가적인 정보는 포함하지 마세요. 예시)"
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = response["choices"][0]["message"]["content"].strip()
        item["thinking"] = explanation
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 환경변수 OPENAI_API_KEY 설정 필요
    main()
