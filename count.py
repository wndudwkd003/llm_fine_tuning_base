import json

def count_long_essays(data):
    """
    서술형 문제에서 답변이 500자 이상인 항목의 개수를 세는 함수

    Args:
        data: JSON 데이터 (리스트 또는 단일 딕셔너리)

    Returns:
        int: 500자 이상인 서술형 답변의 개수
    """
    count = 0

    # 단일 딕셔너리인 경우 리스트로 변환
    if isinstance(data, dict):
        data = [data]

    for item in data:
        # 서술형 문제인지 확인
        if (item.get('input', {}).get('question_type') == '서술형' and
            'output' in item and 'answer' in item['output']):

            answer = item['output']['answer']
            # 답변이 500자 이상인지 확인
            if len(answer) >= 500:
                count += 1

    return count

# 예시 사용법
if __name__ == "__main__":
    # JSON 파일에서 데이터 읽기
    with open('encoded_output/af8b73542c0112d2f37d3513c8222ca0/kakaocorp_kanana-1.5-8b-base_sft_lora_1-3_early_r_64_alpha_64_dropout_0.1_v1_fp16.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 500자 이상인 서술형 답변 개수 계산
    result = count_long_essays(data)
    print(f"500자 이상인 서술형 답변 개수: {result}")

    # 전체 데이터 개수 출력
    print(f"전체 데이터 개수: {len(data)}")

    # 서술형 문제만 필터링해서 답변 글자수 분석
    essay_answers = []
    for item in data:
        if (item.get('input', {}).get('question_type') == '서술형' and
            'output' in item and 'answer' in item['output']):
            essay_answers.append(len(item['output']['answer']))

    if essay_answers:
        print(f"서술형 답변 개수: {len(essay_answers)}")
        print(f"서술형 답변 평균 글자수: {sum(essay_answers)/len(essay_answers):.1f}")
        print(f"서술형 답변 최대 글자수: {max(essay_answers)}")
        print(f"서술형 답변 최소 글자수: {min(essay_answers)}")
    else:
        print("서술형 답변이 없습니다.")

    # 2. JSON 파일에서 읽어오는 경우
    """
    with open('your_file.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = count_long_essays(data)
    print(f"500자 이상인 서술형 답변 개수: {result}")
    """

    # 3. 여러 개의 JSON 객체가 있는 리스트인 경우
    """
    json_list = [
        {
            "id": "591",
            "input": {"question_type": "서술형", ...},
            "output": {"answer": "긴 답변..."},
            ...
        },
        {
            "id": "592",
            "input": {"question_type": "서술형", ...},
            "output": {"answer": "또 다른 답변..."},
            ...
        }
    ]

    result = count_long_essays(json_list)
    print(f"500자 이상인 서술형 답변 개수: {result}")
    """
