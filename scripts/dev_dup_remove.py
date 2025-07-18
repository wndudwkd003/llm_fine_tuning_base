import json
import os
import shutil
from collections import OrderedDict

def remove_duplicates_from_dev_json(source_folder, target_folder):
    """
    dev.json에서 중복 질문 제거하고 전체 폴더를 새 위치에 복사
    """
    # 새 폴더명 생성
    if not target_folder:
        target_folder = source_folder + "_remove_duplication"

    # 기존 폴더가 있으면 삭제
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

    # 전체 폴더 복사
    shutil.copytree(source_folder, target_folder)
    print(f"폴더 복사 완료: {source_folder} -> {target_folder}")

    # dev.json 파일 경로
    dev_json_path = os.path.join(target_folder, "dev.json")

    if not os.path.exists(dev_json_path):
        print(f"dev.json 파일을 찾을 수 없습니다: {dev_json_path}")
        return

    # dev.json 파일 로드
    with open(dev_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"원본 데이터 개수: {len(data)}개")

    # 중복 제거 (question 기준으로 첫 번째 항목만 유지)
    seen_questions = set()
    unique_data = []

    for item in data:
        question = item['input']['question']
        if question not in seen_questions:
            seen_questions.add(question)
            unique_data.append(item)

    print(f"중복 제거 후 데이터 개수: {len(unique_data)}개")
    print(f"제거된 중복 항목: {len(data) - len(unique_data)}개")

    # 결과를 dev.json에 저장
    with open(dev_json_path, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)

    print(f"중복 제거 완료: {dev_json_path}")

# 실행
source_folder = "datasets/merged_dataset_no_aug_v1-3-cot"
target_folder = source_folder + "_remove_duplication"

remove_duplicates_from_dev_json(source_folder, target_folder)
