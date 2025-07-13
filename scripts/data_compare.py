import json
import os

def compare_json_files(path_1, path_2):
    """두 JSON 파일이 동일한지 비교합니다."""
    try:
        with open(path_1, 'r', encoding='utf-8') as file_1, open(path_2, 'r', encoding='utf-8') as file_2:
            data_1 = json.load(file_1)
            data_2 = json.load(file_2)
            return data_1 == data_2
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"❌ 파일 비교 오류: {e}")
        return False

def find_matching_json_in_encoded_output(target_json_path):
    """
    target_json_path와 동일한 JSON 파일을 encoded_output의 해시 폴더들에서 찾습니다.

    Args:
        target_json_path (str): 비교할 기준 JSON 파일 경로

    Returns:
        list: 매칭되는 해시 폴더들의 절대 경로 리스트
    """
    encoded_output_dir = "encoded_output"
    matching_folders = []

    # 기준 파일이 존재하는지 확인
    if not os.path.exists(target_json_path):
        print(f"❌ 기준 파일이 존재하지 않습니다: {target_json_path}")
        return matching_folders

    # encoded_output 폴더가 존재하는지 확인
    if not os.path.exists(encoded_output_dir):
        print(f"❌ {encoded_output_dir} 폴더가 존재하지 않습니다.")
        return matching_folders

    print(f"🔍 {target_json_path} 파일과 동일한 JSON을 찾는 중...")
    print("-" * 50)

    # 모든 해시 폴더 탐색
    for hash_folder in os.listdir(encoded_output_dir):
        hash_path = os.path.join(encoded_output_dir, hash_folder)

        # 폴더인지 확인
        if os.path.isdir(hash_path):
            # 해시 폴더 내의 모든 JSON 파일 확인
            for file_name in os.listdir(hash_path):
                if file_name.endswith('.json'):
                    json_file_path = os.path.join(hash_path, file_name)

                    # JSON 파일 비교
                    if compare_json_files(target_json_path, json_file_path):
                        absolute_path = os.path.abspath(hash_path)
                        matching_folders.append(absolute_path)
                        print(f"✅ 매칭: {absolute_path}")

                        # 원본 정보도 출력 (있다면)
                        original_path_file = os.path.join(hash_path, "txt", "original_path.txt")
                        if os.path.exists(original_path_file):
                            try:
                                with open(original_path_file, 'r', encoding='utf-8') as f:
                                    original_path = f.read().strip()
                                folder_name = original_path.split('/')[-3] if len(original_path.split('/')) >= 3 else "알 수 없음"
                                print(f"   📂 원본 폴더: {folder_name}")
                            except Exception:
                                pass
                        print("-" * 30)
                        break  # 해당 해시 폴더에서 매칭되는 JSON을 찾았으므로 다음 폴더로

    return matching_folders

def main():
    """메인 함수"""
    target_json_path = "encoded_output/9326.json"

    print("🚀 JSON 파일 비교 도구")
    print("=" * 50)

    matching_folders = find_matching_json_in_encoded_output(target_json_path)

    if matching_folders:
        print(f"\n🎉 총 {len(matching_folders)}개의 매칭되는 해시 폴더를 찾았습니다:")
        for i, folder_path in enumerate(matching_folders, 1):
            print(f"{i}. {folder_path}")
    else:
        print(f"\n❌ {target_json_path}와 동일한 JSON 파일을 찾을 수 없습니다.")
        print("💡 가능한 원인:")
        print("   - 해당 JSON 파일이 처리되지 않았음")
        print("   - 파일 내용이 다름")
        print("   - encoded_output 폴더가 비어있음")

if __name__ == "__main__":
    main()
