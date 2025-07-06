import os

def find_sentences_with_word(directories, target_word):
    """
    지정된 디렉토리 목록 내의 .dev, .test, .train 파일에서
    특정 단어가 포함된 문장을 찾아 출력합니다.
    """
    file_endings = ('.dev', '.test', '.train')

    for directory in directories:
        if not os.path.isdir(directory):
            print(f"경고: '{directory}' 디렉토리를 찾을 수 없습니다.")
            continue

        for filename in os.listdir(directory):
            if filename.endswith(file_endings):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if target_word in line:
                                print(f"[{filepath}] {line.strip()}")
                except Exception as e:
                    print(f"'{filepath}' 파일 처리 중 오류 발생: {e}")

# --- 실행 코드 ---
kowiki = "datasets/kowikitext"
namuwiki = "datasets/namuwikitext"
search_word = "해저터널"  # 찾고 싶은 단어를 여기에 입력하세요.

directories_to_search = [kowiki, namuwiki]
find_sentences_with_word(directories_to_search, search_word)
