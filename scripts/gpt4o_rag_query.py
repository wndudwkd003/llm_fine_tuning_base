import os, json, shutil, yaml, time
from openai import OpenAI, RateLimitError

DATA_PATH = "datasets/merged_dataset_no_aug_v1-3"
TARGET_PATH = DATA_PATH + "_rag_queries_v2"
TARGET_FILE = {"train.json": True, "dev.json": True, "test.json": False}
YAML_PATH = "src/configs/token.yaml"
USING_MODEL = "gpt-4.1-2025-04-14"

def get_token():
    with open(YAML_PATH, "r") as f:
        return yaml.safe_load(f)["open_ai_token"]

def make_prompt(sample):
    d_input = sample["input"]
    question = d_input["question"]
    answer = sample["output"]["answer"]
    question_type = d_input.get("question_type", "")

    other_info = ", ".join([
        "카테고리: " + d_input.get("category", ""),
        "도메인: " + d_input.get("domain", ""),
        "문제 타입: " + question_type,
        "키워드: " + d_input.get("topic_keyword", "")
    ])

    base = ("[필수지시사항] 지금부터 당신은 주어진 [질문]과 [기타정보]를 분석하여 RAG(Retrieval-Augmented Generation) 시스템에서 "
            "관련 정보를 효과적으로 검색할 수 있는 질의를 생성해야 합니다. "
            "마크다운·불필요한 구두점·장식 문구·줄바꿈 문자('\\n')를 사용하지 마십시오.")

    # 단답형과 서술형에 따른 지시사항 구분
    if question_type == "단답형" or any(keyword in question_type for keyword in ["단답형", "선다형"]):
        instruction = ("[지시사항] 단답형 문제를 위한 3가지 유형의 검색 질의를 생성하십시오: "
                      "1) 핵심 키워드 기반 검색 질의 - 질문에 포함된 주요 개념이나 용어의 정의를 찾기 위한 질의 "
                      "2) 사실 관계 검색 질의 - 질문과 관련된 구체적인 사실이나 데이터를 찾기 위한 질의 "
                      "3) 맥락 정보 검색 질의 - 질문의 배경이 되는 관련 정보나 연관 개념을 찾기 위한 질의")
    else:
        instruction = ("[지시사항] 서술형 문제를 위한 3가지 유형의 검색 질의를 생성하십시오: "
                      "1) 핵심 개념 검색 질의 - 질문에서 다루는 주요 개념이나 이론의 설명을 찾기 위한 질의 "
                      "2) 사실 관계 검색 질의 - 질문과 관련된 구체적인 사실, 원인, 결과, 과정을 찾기 위한 질의 "
                      "3) 배경 지식 검색 질의 - 질문을 이해하고 답변하는데 필요한 배경 정보나 맥락을 찾기 위한 질의")

    output_format = ("각 질의는 <검색></검색> 태그로 감싸서 출력하십시오. "
                    "예시: <검색>첫 번째 검색 질의</검색> <검색>두 번째 검색 질의</검색> <검색>세 번째 검색 질의</검색>")

    return f"{base} [기타정보] {other_info} 당신이 분석할 [질문]은 {question} 입니다. {instruction} {output_format}"

def save_progress(data, filepath):
    """진행 상황을 즉시 저장하는 함수"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def extract_search_queries(response_text):
    """응답에서 <검색></검색> 태그로 감싸진 질의들을 추출"""
    import re
    pattern = r'<검색>(.*?)</검색>'
    queries = re.findall(pattern, response_text, re.DOTALL)
    return [query.strip() for query in queries]

def load_existing_data(output_path):
    """기존 처리된 데이터를 로드하는 함수"""
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def refine_data_with_immediate_save(data, client: OpenAI, output_path):
    """응답 받을 때마다 즉시 저장하는 버전"""
    # 기존 처리된 데이터 로드
    existing_data = load_existing_data(output_path)
    existing_count = len(existing_data)

    # 기존 데이터가 있으면 해당 부분부터 시작
    if existing_count > 0:
        print(f"기존 처리된 데이터 {existing_count}개 발견, {existing_count + 1}번부터 시작")
        refined = existing_data
        start_idx = existing_count
    else:
        refined = []
        start_idx = 0

    for idx in range(start_idx, len(data)):
        sample = data[idx]
        prompt = make_prompt(sample)
        messages = [
            {"role": "system", "content": "당신은 한국 문화·역사·과학기술 분야의 전문가로서 RAG 시스템을 위한 효과적인 검색 질의를 생성하는 전문가입니다. 주어진 질문과 키워드만을 바탕으로 답변을 찾는데 필요한 정보를 검색할 수 있는 질의를 만들어야 합니다."},
            {"role": "user", "content": prompt}
        ]

        try:
            rsp = client.chat.completions.create(
                model=USING_MODEL,
                messages=messages,
                max_tokens=1024,
                n=1,
                temperature=0.7
            )

            response_text = rsp.choices[0].message.content.strip()
            search_queries = extract_search_queries(response_text)

            # 검색 질의가 3개 미만인 경우 경고
            if len(search_queries) < 3:
                print(f"경고: {idx+1}번 샘플에서 검색 질의가 {len(search_queries)}개만 생성됨")
                # 빈 문자열로 채워서 3개 맞춤
                while len(search_queries) < 3:
                    search_queries.append("")

            sample["output"]["rag_queries"] = search_queries[:3]  # 최대 3개까지만
            sample["output"]["raw_response"] = response_text  # 원본 응답도 저장

            print(f"완료: {idx+1}/{len(data)} - 검색 질의 {len(search_queries)}개 생성")

        except RateLimitError as e:
            print(f"Rate-limit 오류: {e}")
            sample["output"]["rag_queries"] = []
            sample["output"]["raw_response"] = ""
        except Exception as e:
            print(f"처리 오류 {idx+1}: {e}")
            sample["output"]["rag_queries"] = []
            sample["output"]["raw_response"] = ""

        refined.append(sample)

        # 매 요청마다 즉시 저장
        save_progress(refined, output_path)
        print(f"저장 완료: {output_path} ({idx+1}/{len(data)} 샘플)")

        # 속도 제한 방지
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

        # 기존 처리된 데이터 확인
        existing_data = load_existing_data(dst)
        if len(existing_data) == len(data):
            print(f"이미 모든 샘플이 처리되었습니다: {fname}")
            continue
        elif len(existing_data) > 0:
            print(f"기존 처리된 샘플 {len(existing_data)}개 발견, 나머지 {len(data) - len(existing_data)}개 처리 예정")

        refined = refine_data_with_immediate_save(data, client, dst)
        print(f"최종 완료: {fname}")

        # 통계 출력
        total_samples = len(refined)
        samples_with_queries = sum(1 for sample in refined if sample["output"].get("rag_queries"))
        print(f"통계: 전체 {total_samples}개 샘플 중 {samples_with_queries}개 샘플에서 검색 질의 생성 성공")
