import os, json, shutil, yaml, time
from openai import OpenAI, RateLimitError

DATA_PATH = "datasets/sub_3_data_korean_culture_qa_V1.0_preprocessed"
TARGET_PATH = DATA_PATH + "_cot_refined"
TARGET_FILE = {"train.json": True, "dev.json": True, "test.json": False}
YAML_PATH = "src/configs/token.yaml"
USING_MODEL = "gpt-4.1-mini-2025-04-14"

def get_token():
    with open(YAML_PATH, "r") as f:
        return yaml.safe_load(f)["open_ai_token"]

def make_prompt(sample):
    d_input = sample["input"]
    question = d_input["question"]
    answer = sample["output"]["answer"]
    other_info = ", ".join([
        "카테고리: " + d_input.get("category", ""),
        "도메인: " + d_input.get("domain", ""),
        "문제 타입: " + d_input.get("question_type", ""),
        "키워드: " + d_input.get("topic_keyword", "")
    ])
    base = ("[필수지사항] 지금부터 당신은 [질문]에 대한 실제 정답인 [답변]을 말하지 말고, "
            "[답변]이 도출되는 논리적 추론 과정만 한국어 높임말로 서술하십시오. "
            "마크다운·불필요한 구두점·장식 문구를 사용하지 마십시오.")
    if d_input.get("question_type") == "선다형":
        add = ("[지시사항] 각 보기를 1번부터 4번까지 검토하여 왜 특정 보기가 정답이 되는지 "
               "논리적으로 설명하는 과정만 작성하십시오.")
    else:
        add = ("[지시사항] 문제의 키워드를 바탕으로 관련 배경지식을 서술한 뒤, "
               "답이 도출되는 논리적 연결고리만 설명하십시오.")
    return f"{base}\n[기타정보] {other_info}\n당신이 알아야할 [질문]은 {question} 입니다. 그리고 [답변]은 {answer} 입니다. \n{add}"

def refine_data(data, client: OpenAI):
    refined = []
    for sample in data:
        prompt = make_prompt(sample)
        messages = [
            {"role": "system", "content": "당신은 한국 문화·역사·과학기술 분야 문제를 논리적으로 해결하는 탁월한 전문가입니다. 반드시 한국어 높임말로 서술하고 마크다운 문법을 사용하지 마십시오."},
            {"role": "user", "content": prompt}
        ]
        try:
            rsp = client.chat.completions.create(
                model=USING_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                n=2
            )
            reasonings = [c.message.content.strip() for c in rsp.choices]
            answer = sample["output"]["answer"]
            combined = [f"<추론>{r}</추론><답변>{answer}</답변>" for r in reasonings]
            sample["output"]["cot_answer"] = combined
            print(combined)

        except RateLimitError as e:
            print(f"Rate-limit 오류: {e}")
            sample["output"]["cot_answer"] = []
        refined.append(sample)
        if len(refined) % 20 == 0:
            time.sleep(2)
    return refined

if __name__ == "__main__":
    os.makedirs(TARGET_PATH, exist_ok=True)
    client = OpenAI(api_key=get_token())
    for fname, do_refine in TARGET_FILE.items():
        src, dst = os.path.join(DATA_PATH, fname), os.path.join(TARGET_PATH, fname)
        if not do_refine:
            shutil.copy(src, dst); continue
        with open(src, "r", encoding="utf-8") as f:
            refined = refine_data(json.load(f), client)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(refined, f, ensure_ascii=False, indent=2)
