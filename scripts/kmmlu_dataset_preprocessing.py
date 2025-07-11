import os, json
from glob import glob
from sklearn.model_selection import train_test_split


# category_translations = {
#     "Accounting": "회계학",
#     "Agricultural Sciences": "농학",
#     "Aviation Engineering and Maintenance": "항공공학 및 정비",
#     "Biology": "생물학",
#     "Chemical Engineering": "화학공학",
#     "Chemistry": "화학",
#     "Civil Engineering": "토목공학",
#     "Computer Science": "컴퓨터과학",
#     "Construction": "건설학",
#     "Criminal Law": "형법",
#     "Ecology": "생태학",
#     "Economics": "경제학",
#     "Education": "교육학",
#     "Electrical Engineering": "전기공학",
#     "Electronics Engineering": "전자공학",
#     "Energy Management": "에너지관리",
#     "Environmental Science": "환경과학",
#     "Fashion": "패션학",
#     "Food Processing": "식품가공학",
#     "Gas Technology and Engineering": "가스기술 및 공학",
#     "Geomatics": "지리정보과학",
#     "Health": "보건학",
#     "Industrial Engineer": "산업공학",
#     "Information Technology": "정보기술",
#     "Interior Architecture and Design": "실내건축 및 디자인",
#     "Korean History": "한국사",
#     "Law": "법학",
#     "Math": "수학",
#     "Machine Design and Manufacturing": "기계설계 및 제조",
#     "Management": "경영학",
#     "Maritime Engineering": "해양공학",
#     "Marketing": "마케팅학",
#     "Materials Engineering": "재료공학",
#     "Mechanical Engineering": "기계공학",
#     "Nondestructive Testing": "비파괴검사",
#     "Patent": "특허학",
#     "Political Science and Sociology": "정치학 및 사회학",
#     "Psychology": "심리학",
#     "Public Safety": "공공안전학",
#     "Railway and Automotive Engineering": "철도 및 자동차공학",
#     "Real Estate": "부동산학",
#     "Refrigerating Machinery": "냉동기계",
#     "Social Welfare": "사회복지학",
#     "Taxation": "조세학",
#     "Telecommunications and Wireless Technology": "통신 및 무선기술"
# }



category_translations = {
    "Korean History": "한국사",  # 한국 역사 - 직접 관련
    "Political Science and Sociology": "정치학 및 사회학",  # 사회 분야 - 직접 관련
    "Social Welfare": "사회복지학",  # 사회 분야 - 직접 관련
    "Psychology": "심리학",  # 사회과학 분야 - 관련
    "Economics": "경제학",  # 사회과학 분야 - 관련
    "Electrical Engineering": "전기공학",  # 과학기술 분야 - 직접 관련
    "Mechanical Engineering": "기계공학",  # 과학기술 분야 - 직접 관련
    "Materials Engineering": "재료공학",  # 과학기술 분야 - 직접 관련
    "Chemical Engineering": "화학공학",  # 과학기술 분야 - 직접 관련
    "Environmental Science": "환경과학",  # 과학기술 분야 - 직접 관련
    "Information Technology": "정보기술",  # 과학기술 분야 - 직접 관련
    "Computer Science": "컴퓨터과학",  # 과학기술 분야 - 직접 관련
    "Telecommunications and Wireless Technology": "통신 및 무선기술",  # 과학기술 분야 - 직접 관련
    "Math": "수학"  # 과학기술의 기초 분야 - 직접 관련
}


def convert_to_dataset_format(item, category, idx):
    question_text = item["question"]
    choices = [item["A"], item["B"], item["C"], item["D"]]
    answer_index = item["answer"]
    category_en = item.get("Category", category)

    # 번역된 카테고리 이름 가져오기 (없으면 빈 문자열)
    category_ko = category_translations.get(category_en, "")

    choice_str = " ".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
    full_question = f"{question_text} {choice_str}"

    return {
        "id": f"{category_en}_{idx}",
        "input": {
            "category": category_ko,
            "domain": "",
            "question_type": "선다형",
            "topic_keyword": "",
            "question": full_question
        },
        "output": {
            "answer": str(answer_index)
        }
    }


if __name__ == "__main__":
    data_dir = "datasets/KMMLU"
    target_dir = data_dir + "_converted"
    os.makedirs(target_dir, exist_ok=True)

    merged_data = []
    category_set = set()
    untranslated_categories = set()

    # math 관련 test/train/dev 제외 모든 파일 로드
    file_list = [
        f for f in glob(os.path.join(data_dir, "*.json"))
        if not any(skip in os.path.basename(f) for skip in ["test"])
    ]

    for file_path in file_list:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            category_from_filename = os.path.splitext(os.path.basename(file_path))[0]
            for idx, item in enumerate(data):
                category_en = item.get("Category", category_from_filename)

                # ✅ category_translations에 없는 카테고리는 제외
                if category_en not in category_translations:
                    untranslated_categories.add(category_en)
                    continue

                category_ko = category_translations[category_en]
                category_set.add(category_en)
                merged_data.append(
                    convert_to_dataset_format(item, category_en, idx)
                )

    # 중복 제거된 카테고리 출력
    print("\n[사용된 Category 목록]")
    for cat in sorted(category_set):
        print(cat)

    # 전체 데이터를 변환 (분할하지 않음)
    with open(os.path.join(target_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"train.json 저장 완료 ({len(merged_data)}개)")

    # 빈 dev.json 파일 생성
    with open(os.path.join(target_dir, "dev.json"), "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
        print(f"dev.json 저장 완료 (0개)")

    # 마지막에 출력
    if untranslated_categories:
        print("\n[번역되지 않아 제외된 카테고리]")
        for cat in sorted(untranslated_categories):
            print(cat)
