import json
from typing import Dict, Any, List, Tuple

def compare_json_simple(json1: Dict[Any, Any], json2: Dict[Any, Any]) -> bool:
    """
    두 JSON 객체가 완전히 동일한지 간단하게 비교
    """
    return json1 == json2

def compare_json_detailed(json1: Dict[Any, Any], json2: Dict[Any, Any]) -> Tuple[bool, List[str]]:
    """
    두 JSON 객체를 상세하게 비교하고 차이점을 반환
    """
    differences = []

    def compare_recursive(obj1, obj2, path=""):
        if type(obj1) != type(obj2):
            differences.append(f"타입 불일치 at {path}: {type(obj1).__name__} vs {type(obj2).__name__}")
            return

        if isinstance(obj1, dict):
            # 키 비교
            keys1 = set(obj1.keys())
            keys2 = set(obj2.keys())

            # 누락된 키
            missing_in_json2 = keys1 - keys2
            missing_in_json1 = keys2 - keys1

            for key in missing_in_json2:
                differences.append(f"키 누락 in JSON2 at {path}: '{key}'")

            for key in missing_in_json1:
                differences.append(f"키 누락 in JSON1 at {path}: '{key}'")

            # 공통 키의 값 비교
            common_keys = keys1 & keys2
            for key in common_keys:
                new_path = f"{path}.{key}" if path else str(key)
                compare_recursive(obj1[key], obj2[key], new_path)

        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                differences.append(f"배열 길이 불일치 at {path}: {len(obj1)} vs {len(obj2)}")
                return

            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                compare_recursive(item1, item2, new_path)

        else:
            if obj1 != obj2:
                differences.append(f"값 불일치 at {path}: '{obj1}' vs '{obj2}'")

    compare_recursive(json1, json2)
    return len(differences) == 0, differences

def compare_json_ignore_order(json1: Dict[Any, Any], json2: Dict[Any, Any]) -> bool:
    """
    순서를 무시하고 JSON 객체 비교 (딕셔너리는 기본적으로 순서 무시, 리스트는 정렬 후 비교)
    """
    def normalize_json(obj):
        if isinstance(obj, dict):
            return {k: normalize_json(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            # 리스트 요소가 정렬 가능한 경우에만 정렬
            try:
                return sorted([normalize_json(item) for item in obj])
            except TypeError:
                # 정렬할 수 없는 경우 원래 순서 유지
                return [normalize_json(item) for item in obj]
        else:
            return obj

    return normalize_json(json1) == normalize_json(json2)

def load_and_compare_json_files(file1_path: str, file2_path: str) -> None:
    """
    파일에서 JSON을 로드하여 비교
    """
    try:
        print("파일 로딩 중...")
        with open(file1_path, 'r', encoding='utf-8') as f1:
            json1 = json.load(f1)
        print(f"파일1 로딩 완료: {len(json1) if isinstance(json1, (list, dict)) else 1}개 항목")

        with open(file2_path, 'r', encoding='utf-8') as f2:
            json2 = json.load(f2)
        print(f"파일2 로딩 완료: {len(json2) if isinstance(json2, (list, dict)) else 1}개 항목")

        print("\n=== JSON 파일 비교 결과 ===")
        print(f"파일1: {file1_path.split('/')[-1]}")  # 파일명만 표시
        print(f"파일2: {file2_path.split('/')[-1]}")

        # 간단한 비교
        is_equal = compare_json_simple(json1, json2)
        print(f"\n✅ 간단한 비교 결과: {'동일' if is_equal else '다름'}")

        # 상세한 비교
        is_equal_detailed, differences = compare_json_detailed(json1, json2)
        print(f"📋 상세한 비교 결과: {'동일' if is_equal_detailed else '다름'}")

        if not is_equal_detailed:
            print(f"\n❌ 총 {len(differences)}개의 차이점 발견:")
            for i, diff in enumerate(differences[:20], 1):  # 최대 20개까지 표시
                print(f"  {i:2d}. {diff}")
            if len(differences) > 20:
                print(f"     ... 그리고 {len(differences) - 20}개 더")
        else:
            print("\n✅ 두 JSON 파일이 완전히 동일합니다!")

        # 파일 크기 정보 추가
        import os
        size1 = os.path.getsize(file1_path)
        size2 = os.path.getsize(file2_path)
        print(f"\n📁 파일 크기:")
        print(f"   파일1: {size1:,} bytes")
        print(f"   파일2: {size2:,} bytes")
        if size1 != size2:
            print(f"   크기 차이: {abs(size1-size2):,} bytes")

    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        print("경로를 다시 확인해주세요.")
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        print("파일이 올바른 JSON 형식인지 확인해주세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")

def get_file_statistics(file_path: str) -> None:
    """
    JSON 파일의 통계 정보를 출력
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\n📊 {file_path.split('/')[-1]} 통계:")

        if isinstance(data, list):
            print(f"   - 타입: 리스트")
            print(f"   - 항목 수: {len(data):,}개")
            if data and isinstance(data[0], dict):
                print(f"   - 첫 번째 항목의 키: {list(data[0].keys())}")
        elif isinstance(data, dict):
            print(f"   - 타입: 딕셔너리")
            print(f"   - 키 개수: {len(data):,}개")
            print(f"   - 키들: {list(data.keys())[:5]}{'...' if len(data) > 5 else ''}")
        else:
            print(f"   - 타입: {type(data).__name__}")

    except Exception as e:
        print(f"❌ 통계 정보 생성 오류: {e}")

# 실제 파일 비교 실행
if __name__ == "__main__":
    # 비교할 파일 경로
    file1 = "output/kakaocorp_kanana-1.5-8b-base_sft_lora_1-3_early_r_64_alpha_64_dropout_0.1_v1_1_fp16/pred_result/kakaocorp_kanana-1.5-8b-base_sft_lora_1-3_early_r_64_alpha_64_dropout_0.1_v1_1_fp16.json"
    file2 = "output/kakaocorp_kanana-1.5-8b-base_sft_lora_1-3_early_r_64_alpha_64_dropout_0.1_v1_1_fp16/pred_result_temp/kakaocorp_kanana-1.5-8b-base_sft_lora_1-3_early_r_64_alpha_64_dropout_0.1_v1_1_fp16.json"

    print("🔍 JSON 파일 비교 시작...")

    # 각 파일의 통계 정보 출력
    get_file_statistics(file1)
    get_file_statistics(file2)

    # 파일 비교 수행
    load_and_compare_json_files(file1, file2)
