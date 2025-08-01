import os
import shutil
import hashlib
import json
from pathlib import Path

def create_hash_from_path(file_path):
    """파일 경로를 해시로 변환합니다."""
    return hashlib.md5(file_path.encode('utf-8')).hexdigest()

def organize_output_files():
    """
    output 폴더 내의 모든 pred_result JSON 파일들을 해시 기반 폴더로 정리해서 encoded_output에 저장합니다.
    """
    target_dir = "encoded_output"
    output_dir = "output"

    # target_dir 생성
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"✅ {target_dir} 폴더가 생성되었습니다.")

    # output 폴더가 존재하는지 확인
    if not os.path.exists(output_dir):
        print(f"❌ {output_dir} 폴더가 존재하지 않습니다.")
        return

    processed_count = 0

    # output 폴더 내의 모든 하위 폴더 탐색
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)

        # 폴더인지 확인
        if os.path.isdir(item_path):
            # pred_result 폴더 경로
            pred_result_path = os.path.join(item_path, "pred_result")

            # pred_result 폴더가 존재하는지 확인
            if os.path.exists(pred_result_path):
                # JSON 파일 찾기
                json_files = [f for f in os.listdir(pred_result_path) if f.endswith('.json')]

                for json_file in json_files:
                    json_path = os.path.join(pred_result_path, json_file)

                    # 해시 생성 (원본 경로 기준)
                    hash_name = create_hash_from_path(json_path)

                    # 해시 폴더 생성
                    hash_folder = os.path.join(target_dir, hash_name)
                    if not os.path.exists(hash_folder):
                        os.makedirs(hash_folder)

                    # JSON 파일 복사
                    target_json_path = os.path.join(hash_folder, json_file)
                    shutil.copy2(json_path, target_json_path)
                    print(f"📁 JSON 복사: {json_path} -> {target_json_path}")

                    # txt 폴더 생성 및 원래 경로 저장
                    txt_folder = os.path.join(hash_folder, "txt")
                    if not os.path.exists(txt_folder):
                        os.makedirs(txt_folder)

                    # 원래 경로를 txt 파일에 저장
                    original_path_file = os.path.join(txt_folder, "original_path.txt")
                    with open(original_path_file, 'w', encoding='utf-8') as f:
                        f.write(json_path)

                    # config.py 파일 복사
                    config_path = os.path.join(item_path, "backup", "config.py")
                    if os.path.exists(config_path):
                        target_config_path = os.path.join(hash_folder, "config.py")
                        shutil.copy2(config_path, target_config_path)
                        print(f"⚙️  Config 복사: {config_path} -> {target_config_path}")
                    else:
                        print(f"⚠️  Config 파일을 찾을 수 없습니다: {config_path}")

                    processed_count += 1
                    print(f"✅ 처리 완료: {hash_name} (원본: {item})")
                    print("-" * 50)
            else:
                print(f"⚠️  pred_result 폴더가 없습니다: {pred_result_path}")

    print(f"\n🎉 총 {processed_count}개의 파일이 처리되었습니다.")

def create_summary_report():
    """처리된 파일들의 요약 보고서를 생성합니다."""
    target_dir = "encoded_output"

    if not os.path.exists(target_dir):
        print(f"❌ {target_dir} 폴더가 존재하지 않습니다.")
        return

    summary_data = []

    # 각 해시 폴더 정보 수집
    for hash_folder in os.listdir(target_dir):
        hash_path = os.path.join(target_dir, hash_folder)
        if os.path.isdir(hash_path):
            # 원본 경로 읽기
            original_path_file = os.path.join(hash_path, "txt", "original_path.txt")
            if os.path.exists(original_path_file):
                with open(original_path_file, 'r', encoding='utf-8') as f:
                    original_path = f.read().strip()

                # JSON 파일 확인
                json_files = [f for f in os.listdir(hash_path) if f.endswith('.json')]

                # config.py 파일 확인
                config_exists = os.path.exists(os.path.join(hash_path, "config.py"))

                summary_data.append({
                    "hash": hash_folder,
                    "original_path": original_path,
                    "json_files": json_files,
                    "config_exists": config_exists
                })

    # 요약 보고서 저장
    summary_file = os.path.join(target_dir, "summary_report.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"📊 요약 보고서가 생성되었습니다: {summary_file}")

    # 콘솔에 요약 정보 출력
    print(f"\n📋 처리된 파일 요약:")
    for item in summary_data:
        print(f"  해시: {item['hash'][:12]}...")
        print(f"  원본: {item['original_path']}")
        print(f"  JSON: {', '.join(item['json_files'])}")
        print(f"  Config: {'✅' if item['config_exists'] else '❌'}")
        print("-" * 40)

def verify_no_duplicates():
    """중복이 없는지 확인합니다."""
    target_dir = "encoded_output"

    if not os.path.exists(target_dir):
        print(f"❌ {target_dir} 폴더가 존재하지 않습니다.")
        return

    hash_folders = [f for f in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, f))]

    if len(hash_folders) == len(set(hash_folders)):
        print(f"✅ 중복 없음: {len(hash_folders)}개의 고유한 해시 폴더가 생성되었습니다.")
    else:
        print(f"❌ 중복 발견: 총 {len(hash_folders)}개 중 고유한 것은 {len(set(hash_folders))}개입니다.")

if __name__ == "__main__":
    print("🚀 Output 파일 정리 작업을 시작합니다...")
    print("=" * 60)

    # 메인 작업 실행
    organize_output_files()

    # 요약 보고서 생성
    create_summary_report()

    # 중복 검증
    verify_no_duplicates()

    print("\n🏁 모든 작업이 완료되었습니다!")
