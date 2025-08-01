import os, json, shutil, yaml, time
from openai import OpenAI, RateLimitError
import re

DATA_PATH = "datasets/etri_qa_abc"
TARGET_PATH = DATA_PATH + "_cot_3-stage_refined_4.1"
TARGET_FILE = {"train.json": True}
YAML_PATH = "src/configs/token.yaml"
USING_MODEL = "gpt-4.1-2025-04-14"

def get_token():
    with open(YAML_PATH, "r") as f:
        return yaml.safe_load(f)["open_ai_token"]


def extract_reasoning(text: str) -> str:
    """<추론> 태그 안의 내용만 추출"""
    match = re.search(r"<추론>(.*?)</추론>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


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
    base = (
        "[필수지사항] 지금부터 당신은 [질문]에 대한 [답변]을 도출화는 과정에 미묘하지만 엉뚱하고 '틀린' 추론 과정을 출력해야합니다. "
        "[답변]이 도출되는 논리적이지만 '틀린' 추론 과정을 출력해야합니다. 한국어 높임말로 서술해야하며, [질문]이 무엇을 요구하는지 '요지'를 파악하는 것은 중요하지만, [답변]이 도출되기 위한 틀린 연결고리를 찾는것이 중요합니다. "
        "여기서 중요한 것은 출력 형식은 정확해야합니다. <추론>추론내용</추론><답변>답변내용</답변> 형식으로 확실하게 작성해야합니다. "
        "마크다운·불필요한 구두점·장식 문구 줄바꿈 문자('\\n')를 사용하지 마십시오. <답변>답변내용</답변>의 '답변내용'에는 ~입니다. 로 끝나지 않도록 하세요. 선다형은 보기 내용을 반복하지말고 보기의 숫자만, 단답형은 단어만, 서술형은 문장으로 작성해야합니다."
        "'추론내용' 부분은 ~로 잘못선택할 수 있습니다. 로 끝나면 안됩니다. 반드시 정확하게 결론을 지었다고 가정하고 추론내용을 작성하세요. 당신은 당당하게 틀린 내용을 맞다고 주장할 수 있습니다. "
    )

    if d_input.get("question_type") == "선다형":
        add = ("[지시사항] 먼저 [질문]의 요지를 파악하고, 각 보기를 1번부터 4번까지 검토하여 왜 특정 보기가 실제 정답이 아닌 다른 정답을 고르고 정답이라고 가정하고 그 이유를 논리적으로 설명하십시오. 하지만 미묘하게 틀리게 작성하세요.")
    else:
        add = ("[지시사항] [문제]와 연관된 키워드를 바탕으로 관련된 정확한 배경지식을 서술하세요. 하지만 실제 정답이 아니라 틀린 답을 정답으로 가정하고 해당 답으로 도출되는 논리적 연결고리와 답을 작성하십시오.")
    return f"{base} [기타정보] {other_info} 당신이 알아야할 [질문]은 {question} 입니다. 그리고 실제 정답인 [답변]은 {answer} 입니다. 실제 정답이 아닌 미묘하지만 틀린 [답변]을 정답으로 가정하세요. {add} 다시한번 강조하지만 <추론>추론내용</추론><답변>답변내용</답변> 형식으로 출력하세요."

def save_progress(data, filepath):
    """진행 상황을 즉시 저장하는 함수"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def refine_data_with_immediate_save(data, client: OpenAI, output_path):
    """응답 받을 때마다 즉시 저장하고, <추론> 파싱 실패 시 최대 2회 재요청"""
    refined = []

    def get_reasonings(sample, retries=2):
        """최대 2개의 유효한 추론을 얻을 때까지 재시도"""
        valid = []
        failures = []

        prompt = make_prompt(sample)
        messages = [
            {"role": "system", "content": "당신은 한국 문화·역사·과학기술 분야 문제를 논리적으로 해결하는 탁월한 전문가입니다. 반드시 한국어 높임말로 서술하고 마크다운 문법을 사용하지 마십시오."},
            {"role": "user", "content": prompt}
        ]

        # 초기 요청 (n=2)
        try:
            rsp = client.chat.completions.create(
                model=USING_MODEL,
                messages=messages,
                max_tokens=2048,
                n=2
            )
            responses = [c.message.content.strip() for c in rsp.choices]
        except RateLimitError as e:
            print(f"Rate-limit 오류: {e}")
            return [], []

        for raw in responses:
            reasoning = extract_reasoning(raw)
            if reasoning:
                valid.append((reasoning, raw))
            else:
                failures.append(raw)

        # 부족하면 최대 2회 재요청 (n=1)
        retry_count = 0
        while len(valid) < 2 and retry_count < retries:
            try:
                rsp_retry = client.chat.completions.create(
                    model=USING_MODEL,
                    messages=messages,
                    max_tokens=2048,
                    n=1
                )
                raw = rsp_retry.choices[0].message.content.strip()
                reasoning = extract_reasoning(raw)
                if reasoning:
                    valid.append((reasoning, raw))
                else:
                    failures.append(raw)
            except RateLimitError as e:
                print(f"[재시도 {retry_count+1}] Rate-limit 오류: {e}")
                break
            retry_count += 1

        return valid, failures

    for idx, sample in enumerate(data):
        correct_answer = sample["output"]["answer"]
        combined = []
        failure_log = []

        valid_reasonings, failures = get_reasonings(sample)
        failure_log.extend(failures)

        for reasoning, raw in valid_reasonings:
            combined.append(raw)
            combined.append(f"<추론>{reasoning}</추론><답변>{correct_answer}</답변>")

        sample["output"]["cot_answer"] = combined
        sample["output"]["failure_answer"] = failure_log

        print(f"완료: {idx+1}/{len(data)}개 - 저장된 추론 {len(combined)}개 / 실패 {len(failure_log)}개")
        refined.append(sample)

        save_progress(refined, output_path)
        print(f"저장 완료: {output_path} ({idx+1}/{len(data)} 샘플)")

        if (idx + 1) % 20 == 0:
            time.sleep(2)

    return refined


if __name__ == "__main__":
    os.makedirs(TARGET_PATH, exist_ok=True)
    client = OpenAI(api_key=get_token())

    for fname, do_refine in TARGET_FILE.items():
        src, dst = os.path.join(DATA_PATH, fname), os.path.join(TARGET_PATH, fname)

        if not do_refine:
            shutil.copy(src, dst)
            continue

        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"시작: {fname} ({len(data)} 샘플)")
        refined = refine_data_with_immediate_save(data, client, dst)
        print(f"최종 완료: {fname}")
