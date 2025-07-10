import os, json, re, copy   # copy 모듈 추가

data_dir = "datasets/sub_3_data_korean_culture_qa_V1.0"
target_files = ["dev.json", "test.json", "train.json"]
target_dir = "datasets/sub_3_data_korean_culture_qa_V1.0_preprocessed"

quote_pat = re.compile(r'\\"(.*?)\\"')                      # \"텍스트\" 패턴
option_pat = re.compile(r'(^|\s)(\d+)[ \t]+(?=\S)')

def clean_text(text: str) -> str:
    text = text.replace('\\n', ' ').replace('\\t', ' ')
    text = quote_pat.sub(r"'\1'", text)
    text = option_pat.sub(lambda m: f"{m.group(1)}{m.group(2)}. ", text)
    return ' '.join(text.split())


def expand_answers(item: dict) -> list:
    """
    output.answer 에 '#' 로 구분된 후보가 있을 경우
    각 후보별로 아이템을 복제해 반환합니다.
    """
    answer_str = item.get('output', {}).get('answer', '')
    if '#' not in answer_str:
        return [item]

    answers = [a.strip() for a in answer_str.split('#') if a.strip()]
    expanded = []
    for idx, ans in enumerate(answers, 1):
        new_item = copy.deepcopy(item)
        new_item['output']['answer'] = ans
        # id 충돌 방지: 원본 id 뒤에 _n 부여
        new_item['id'] = f"{item['id']}_{idx}"
        expanded.append(new_item)
    return expanded

def process_file(src_path: str, dst_path: str) -> None:
    with open(src_path, encoding='utf-8') as f:
        data = json.load(f)

    processed = []
    for item in data:
        for section in ('input', 'output'):
            if section in item:
                for k, v in item[section].items():
                    if isinstance(v, str):
                        item[section][k] = clean_text(v)
        processed.extend(expand_answers(item))   # 후보 확장

    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    os.makedirs(target_dir, exist_ok=True)
    for fname in target_files:
        src = os.path.join(data_dir, fname)
        dst = os.path.join(target_dir, fname)
        process_file(src, dst)
    print(f"전처리 완료 → {target_dir}")
