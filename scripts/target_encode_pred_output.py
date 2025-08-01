import os
import shutil
import hashlib
import json
from pathlib import Path

def create_hash_from_path(file_path):
    """파일 경로를 해시로 변환합니다."""
    return hashlib.md5(file_path.encode('utf-8')).hexdigest()

def organize_specific_folder(folder_name):
    """
    특정 폴더의 pred_result JSON 파일들을 해시 기반 폴더로 정리해서 encoded_output에 저장합니다.

    Args:
        folder_name (str): 처리할 폴더명
    """
    target_dir = "encoded_output"
    output_dir = "output"

    # target_dir 생성
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"✅ {target_dir} 폴더가 생성되었습니다.")



    # 지정된 폴더 경로
    item_path = os.path.join(output_dir, folder_name)


    processed_count = 0

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
                print(f"{hash_folder}")

            processed_count += 1
    else:
        print(f"⚠️  pred_result 폴더가 없습니다: {pred_result_path}")

    print(f"\n🎉 총 {processed_count}개의 파일이 처리되었습니다.")
    return processed_count

def organize_all_folders():
    """
    output 폴더 내의 모든 하위 폴더를 처리합니다.
    """
    output_dir = "output"

    if not os.path.exists(output_dir):
        print(f"❌ {output_dir} 폴더가 존재하지 않습니다.")
        return

    total_processed = 0

    # output 폴더 내의 모든 하위 폴더 탐색
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)

        # 폴더인지 확인
        if os.path.isdir(item_path):
            print(f"\n📂 처리 중: {item}")
            print("=" * 40)
            count = organize_specific_folder(item)
            total_processed += count

    print(f"\n🏁 전체 처리 완료: 총 {total_processed}개의 파일이 처리되었습니다.")

def update_summary_report():
    """처리된 파일들의 요약 보고서를 업데이트합니다. (기존 내용에 추가)"""
    target_dir = "encoded_output"

    if not os.path.exists(target_dir):
        print(f"❌ {target_dir} 폴더가 존재하지 않습니다.")
        return

    summary_file = os.path.join(target_dir, "summary_report.json")

    # 기존 데이터 로드
    existing_data = []
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            existing_data = []

    # 기존 해시 목록 생성
    existing_hashes = {item["hash"] for item in existing_data}

    new_data = []

    # 각 해시 폴더 정보 수집
    for hash_folder in os.listdir(target_dir):
        hash_path = os.path.join(target_dir, hash_folder)
        if os.path.isdir(hash_path) and hash_folder not in existing_hashes:
            # 원본 경로 읽기
            original_path_file = os.path.join(hash_path, "txt", "original_path.txt")
            if os.path.exists(original_path_file):
                with open(original_path_file, 'r', encoding='utf-8') as f:
                    original_path = f.read().strip()

                # JSON 파일 확인
                json_files = [f for f in os.listdir(hash_path) if f.endswith('.json')]

                # config.py 파일 확인
                config_exists = os.path.exists(os.path.join(hash_path, "config.py"))

                new_data.append({
                    "hash": hash_folder,
                    "original_path": original_path,
                    "json_files": json_files,
                    "config_exists": config_exists
                })

    # 기존 데이터에 새 데이터 추가
    all_data = existing_data + new_data

    # 요약 보고서 저장
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"📊 요약 보고서가 업데이트되었습니다: {summary_file}")
    print(f"   기존: {len(existing_data)}개, 새로 추가: {len(new_data)}개, 총: {len(all_data)}개")

    # 새로 추가된 항목만 콘솔에 출력


def find_folder_by_name(folder_name):
    """
    폴더명으로 해당하는 해시 폴더를 찾습니다.

    Args:
        folder_name (str): 찾을 폴더명

    Returns:
        str or None: 해시 폴더명 또는 None
    """
    target_dir = "encoded_output"

    if not os.path.exists(target_dir):
        print(f"❌ {target_dir} 폴더가 존재하지 않습니다.")
        return None

    # summary_report.json에서 찾기
    summary_file = os.path.join(target_dir, "summary_report.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                original_path = item["original_path"]
                # 경로에서 폴더명 추출
                if folder_name in original_path:
                    return item["hash"]

        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️  Summary report 읽기 실패: {e}")

    # summary_report가 없거나 실패한 경우 직접 탐색

    for hash_folder in os.listdir(target_dir):
        hash_path = os.path.join(target_dir, hash_folder)
        if os.path.isdir(hash_path):
            original_path_file = os.path.join(hash_path, "txt", "original_path.txt")
            if os.path.exists(original_path_file):
                try:
                    with open(original_path_file, 'r', encoding='utf-8') as f:
                        original_path = f.read().strip()

                    if folder_name in original_path:
                        return hash_folder
                except Exception as e:
                    continue

    return None

def search_and_display_folder(folder_name):
    """
    폴더명을 검색하고 결과를 표시합니다.

    Args:
        folder_name (str): 검색할 폴더명
    """
    print(f"🔍 '{folder_name}' 폴더를 검색 중...")
    print("-" * 50)

    hash_folder = find_folder_by_name(folder_name)

    if hash_folder:
        target_dir = "encoded_output"
        hash_path = os.path.join(target_dir, hash_folder)

        print(f"✅ 폴더를 찾았습니다!")
        print(f"📁 해시 폴더: {hash_folder}")
        print(f"📂 상대 경로: {hash_path}")
        print(f"📁 절대 경로: {os.path.abspath(hash_path)}")

        # 폴더 내용 표시
        if os.path.exists(hash_path):
            print(f"\n📋 폴더 내용:")
            for item in os.listdir(hash_path):
                item_path = os.path.join(hash_path, item)
                if os.path.isfile(item_path):
                    print(f"  📄 {item}")
                elif os.path.isdir(item_path):
                    print(f"  📁 {item}/")

            # 원본 경로 표시
            original_path_file = os.path.join(hash_path, "txt", "original_path.txt")
            if os.path.exists(original_path_file):
                with open(original_path_file, 'r', encoding='utf-8') as f:
                    original_path = f.read().strip()
                print(f"\n🏠 원본 경로: {original_path}")

    else:
        print(f"❌ '{folder_name}' 폴더를 찾을 수 없습니다.")
        print("💡 가능한 원인:")
        print("   - 폴더명이 정확하지 않음")
        print("   - 아직 처리되지 않은 폴더")
        print("   - encoded_output 폴더가 비어있음")

def find_folder_by_hash(hash_input):
    """
    해시값으로 해당하는 폴더를 찾습니다.

    Args:
        hash_input (str): 찾을 해시값 (부분 해시도 가능)

    Returns:
        list: 매칭되는 해시 폴더들의 리스트
    """
    target_dir = "encoded_output"

    if not os.path.exists(target_dir):
        print(f"❌ {target_dir} 폴더가 존재하지 않습니다.")
        return []

    matching_folders = []

    # 모든 해시 폴더 검색
    for hash_folder in os.listdir(target_dir):
        hash_path = os.path.join(target_dir, hash_folder)
        if os.path.isdir(hash_path):
            # 완전 일치 또는 부분 일치 확인
            if hash_input.lower() in hash_folder.lower():
                matching_folders.append(hash_folder)

    return matching_folders

def search_and_display_hash(hash_input):
    """
    해시값을 검색하고 결과를 표시합니다.

    Args:
        hash_input (str): 검색할 해시값
    """
    print(f"🔍 해시 '{hash_input}'를 검색 중...")
    print("-" * 50)

    matching_folders = find_folder_by_hash(hash_input)

    if matching_folders:
        if len(matching_folders) == 1:
            hash_folder = matching_folders[0]
            target_dir = "encoded_output"
            hash_path = os.path.join(target_dir, hash_folder)

            print(f"✅ 해시 폴더를 찾았습니다!")
            print(f"📁 해시 폴더: {hash_folder}")
            print(f"📂 상대 경로: {hash_path}")
            print(f"📁 절대 경로: {os.path.abspath(hash_path)}")

            # 폴더 내용 표시
            if os.path.exists(hash_path):
                print(f"\n📋 폴더 내용:")
                for item in os.listdir(hash_path):
                    item_path = os.path.join(hash_path, item)
                    if os.path.isfile(item_path):
                        print(f"  📄 {item}")
                    elif os.path.isdir(item_path):
                        print(f"  📁 {item}/")

                # 원본 경로 표시
                original_path_file = os.path.join(hash_path, "txt", "original_path.txt")
                if os.path.exists(original_path_file):
                    with open(original_path_file, 'r', encoding='utf-8') as f:
                        original_path = f.read().strip()
                    print(f"\n🏠 원본 경로: {original_path}")

                    # 원본 폴더명 추출
                    folder_name = original_path.split('/')[-3] if len(original_path.split('/')) >= 3 else "알 수 없음"
                    print(f"📂 원본 폴더명: {folder_name}")
        else:
            print(f"✅ {len(matching_folders)}개의 매칭되는 해시 폴더를 찾았습니다:")
            print("=" * 60)

            target_dir = "encoded_output"
            for i, hash_folder in enumerate(matching_folders, 1):
                hash_path = os.path.join(target_dir, hash_folder)
                print(f"{i}. 해시: {hash_folder}")

                # 원본 경로 표시
                original_path_file = os.path.join(hash_path, "txt", "original_path.txt")
                if os.path.exists(original_path_file):
                    try:
                        with open(original_path_file, 'r', encoding='utf-8') as f:
                            original_path = f.read().strip()
                        folder_name = original_path.split('/')[-3] if len(original_path.split('/')) >= 3 else "알 수 없음"
                        print(f"   📂 원본 폴더: {folder_name}")
                    except Exception as e:
                        print(f"   ⚠️  원본 경로 읽기 실패")

                print("-" * 40)

            print(f"\n💡 정확한 해시를 입력하면 상세 정보를 볼 수 있습니다.")
    else:
        print(f"❌ 해시 '{hash_input}'와 매칭되는 폴더를 찾을 수 없습니다.")
        print("💡 가능한 원인:")
        print("   - 해시값이 정확하지 않음")
        print("   - 해당 폴더가 존재하지 않음")
        print("   - encoded_output 폴더가 비어있음")
        print("\n📋 옵션 4번으로 전체 해시 목록을 확인해보세요.")

def list_all_processed_folders():
    """처리된 모든 폴더 목록을 표시합니다."""
    target_dir = "encoded_output"

    if not os.path.exists(target_dir):
        print(f"❌ {target_dir} 폴더가 존재하지 않습니다.")
        return

    # summary_report.json에서 읽기
    summary_file = os.path.join(target_dir, "summary_report.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"📋 처리된 폴더 목록 ({len(data)}개):")
            print("=" * 60)

            for i, item in enumerate(data, 1):
                original_path = item["original_path"]
                # 경로에서 폴더명 추출
                folder_name = original_path.split('/')[-3] if len(original_path.split('/')) >= 3 else "알 수 없음"

                print(f"{i:3d}. {folder_name}")
                print(f"     해시: {item['hash'][:12]}... (전체: {item['hash']})")
                print(f"     JSON: {', '.join(item['json_files'])}")
                print(f"     Config: {'✅' if item['config_exists'] else '❌'}")
                print("-" * 40)

        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️  Summary report 읽기 실패: {e}")
            print("📁 직접 폴더 목록을 확인합니다...")

            hash_folders = [f for f in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, f))]
            print(f"📋 해시 폴더 목록 ({len(hash_folders)}개):")
            for i, hash_folder in enumerate(hash_folders, 1):
                print(f"{i:3d}. {hash_folder[:12]}... (전체: {hash_folder})")
    else:
        print("📋 Summary report가 없습니다. 직접 폴더를 확인해주세요.")

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

    # 사용자 입력 받기
    print("옵션을 선택하세요:")
    print("1. 특정 폴더 처리")
    print("2. 모든 폴더 처리")
    print("3. 폴더명으로 해시 폴더 찾기")
    print("4. 처리된 모든 폴더 목록 보기")
    print("5. 해시로 폴더 찾기")

    choice = input("선택 (1, 2, 3, 4, 또는 5): ").strip()

    if choice == "1":
        folder_name = input("처리할 폴더명을 입력하세요: ").strip()
        if folder_name:
            organize_specific_folder(folder_name)
            # 요약 보고서 업데이트
            update_summary_report()
            # 중복 검증
            verify_no_duplicates()
        else:
            print("❌ 폴더명이 입력되지 않았습니다.")
    elif choice == "2":
        print("\n📂 모든 폴더를 처리합니다...")
        organize_all_folders()
        # 요약 보고서 업데이트
        update_summary_report()
        # 중복 검증
        verify_no_duplicates()
    elif choice == "3":
        folder_name = input("찾을 폴더명을 입력하세요: ").strip()
        if folder_name:
            search_and_display_folder(folder_name)
        else:
            print("❌ 폴더명이 입력되지 않았습니다.")
    elif choice == "4":
        list_all_processed_folders()
    elif choice == "5":
        hash_input = input("찾을 해시값을 입력하세요 (부분 해시도 가능): ").strip()
        if hash_input:
            search_and_display_hash(hash_input)
        else:
            print("❌ 해시값이 입력되지 않았습니다.")
    else:
        print("❌ 잘못된 선택입니다. 1, 2, 3, 4, 또는 5를 입력해주세요.")
        exit()

    print("\n🏁 모든 작업이 완료되었습니다!")
