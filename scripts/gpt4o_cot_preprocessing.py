import json
import os
import re
from copy import deepcopy
import shutil



def process_and_save(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []
    id_counter = 1

    for item in data:
        if 'output' in item and 'cot_answer' in item['output']:
            for cot in item['output']['cot_answer']:
                new_item = {
                    "id": str(id_counter),
                    "input": deepcopy(item["input"]),
                    "output": {
                        "answer": cot
                    }
                }
                new_data.append(new_item)
                id_counter += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"{os.path.basename(output_path)} 저장 완료 ({len(new_data)}개 항목)")

def main():
    base_path = "datasets/etri_qa_abc_cot_refined_4.1"
    output_path = base_path + "_converted"
    os.makedirs(output_path, exist_ok=True)

    # train.json, dev.json 변환
    for split in ['train', 'dev']:
        input_file = os.path.join(base_path, f"{split}.json")
        output_file = os.path.join(output_path, f"{split}.json")
        if os.path.exists(input_file):
            print(f"{split}.json 처리 중...")
            process_and_save(input_file, output_file)

    # test.json 복사
    test_src = os.path.join(base_path, "test.json")
    test_dst = os.path.join(output_path, "test.json")
    if os.path.exists(test_src):
        shutil.copy(test_src, test_dst)
        print("test.json 복사 완료")

    print("\n모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
