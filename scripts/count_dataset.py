import json
from pathlib import Path
from collections import defaultdict
import math

# ────────────────────────────────────────
# 1. 설정
# ────────────────────────────────────────
JSON_PATH = Path("datasets/sub_3_data_korean_culture_qa_V1.0_preprocessed/test.json")
BIN_SIZE = 10         # 구간 폭(문자 수)
MAX_BAR_WIDTH = 50    # 막대 최대 길이(문자)
MAX_ID_SHOW = None    # 구간별 표시할 id 수 (None이면 모두 출력)

# ────────────────────────────────────────
# 2. 데이터 불러오기 및 길이·id 집계
# ────────────────────────────────────────
with JSON_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

# buckets: {구간 시작값: [길이, id 목록]}
buckets = defaultdict(list)

for sample in data:
    q = sample.get("input", {}).get("question")
    sid = sample.get("id")
    if q is None or sid is None:
        continue

    length = len(q)
    bin_start = (length // BIN_SIZE) * BIN_SIZE
    buckets[bin_start].append((length, sid))

# 길이만 별도 추출
all_lengths = [length for (length, _) in sum(buckets.values(), [])]

print(f"샘플 수: {len(all_lengths)}")
print(f"문자 수 평균: {sum(all_lengths) / len(all_lengths):.1f}\n")

# ────────────────────────────────────────
# 3. 구간별 빈도 및 시각적 막대 길이 계산
# ────────────────────────────────────────
max_count = max(len(v) for v in buckets.values())
scale = max_count / MAX_BAR_WIDTH if max_count > MAX_BAR_WIDTH else 1

# 범위를 0부터 최대 길이까지 순차적으로 출력
print("문자 수 구간 | 빈도 | 막대(hist) | 샘플 id 목록")
print("-" * 80)
upper_bound = (max(all_lengths) // BIN_SIZE + 1) * BIN_SIZE
for start in range(0, upper_bound, BIN_SIZE):
    samples = buckets.get(start, [])
    count = len(samples)
    bar_len = math.ceil(count / scale) if count else 0
    end = start + BIN_SIZE - 1
    bar = "█" * bar_len

    # id 목록 준비
    ids = [sid for (_, sid) in samples]
    if MAX_ID_SHOW is not None and len(ids) > MAX_ID_SHOW:
        shown = ids[:MAX_ID_SHOW]
        ids_str = ", ".join(shown) + f", …(+{len(ids) - MAX_ID_SHOW})"
    else:
        ids_str = ", ".join(ids)

    print(f"{start:4d}-{end:<4d} | {count:4d} | {bar:<{MAX_BAR_WIDTH}} | {ids_str}")
