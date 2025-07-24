import json
import os
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from collections import Counter
import re


TYPE_INSTRUCTIONS = {
                "선다형": (
                    "[지시사항] 질문을 잘 읽고 주어진 보기 중에서 정답을 숫자로만 답변하시오. 문제를 그대로 출력하지 마시오."
                ),
                "서술형": (
                    "[지시사항] 질문을 잘 읽고 300자 ~ 500자 이내로 완성된 서술형으로 답변하세요. 최대한 자세히 적되, 핵심 단어를 놓치지 말고 정확하게 사실만 답하여야합니다. 그리고 문제를 그대로 출력하지 마시오."
                ),
                "단답형": (
                    "[지시사항] 질문을 잘 읽고 단답형으로 답하시오. 문제를 그대로 출력하지 마시오."

                ),
            }

def has_excessive_repetition(text, max_repetition=5):
    """
    텍스트에서 동일한 단어가 max_repetition번 이상 반복되는지 확인

    Args:
        text: 검사할 텍스트
        max_repetition: 허용되는 최대 반복 횟수 (기본값: 5)

    Returns:
        bool: 과도한 반복이 있으면 True, 없으면 False
    """
    # 공백과 특수문자로 단어 분리
    words = re.findall(r'\b\w+\b', text.lower())

    # 단어 빈도 계산
    word_counts = Counter(words)

    # 최대 반복 횟수를 초과하는 단어가 있는지 확인
    for word, count in word_counts.items():
        if count >= max_repetition:
            return True

    return False

class CustomDataset(Dataset):
    def __init__(
            self,
            fname,
            tokenizer,
            use_rag=False,   # RAG 사용 여부
            igonore_index=-100,
            prompt="",
            use_system_prompt=False,
            is_test_and_drop_other_info=False,
            enable_length_filtering=False,  # 길이 필터링 활성화 여부
            use_cot=False,  # CoT 사용 여부 추가
    ):
        self.igonore_index = igonore_index
        self.prompt = prompt
        self.use_system_prompt = use_system_prompt
        self.use_rag = use_rag
        self.is_test_and_drop_other_info = is_test_and_drop_other_info
        self.enable_length_filtering = enable_length_filtering
        self.use_cot = use_cot  # CoT 사용 여부 저장
        self.tokenizer = tokenizer  # 토크나이저 저장

        self.remove_question_count = 400
        self.remove_answer_count = 500
        self.top_k = 2  # 기본값을 5로 변경
        self.min_context_length = 15  # 최소 컨텍스트 길이
        self.max_word_repetition = 5  # 최대 단어 반복 허용 횟수

        # 동적 길이 조정을 위한 임계값들 (토큰 단위)
        # 답변 길이에 따른 예약 토큰 설정
        if self.use_cot:
            self.answer_reserve_tokens = 2500  # CoT: 2000자 ≈ 2500토큰
            print(f"CoT 모드: 답변용 {self.answer_reserve_tokens} 토큰 예약")
        else:
            self.answer_reserve_tokens = 700   # 일반: 550자 ≈ 700토큰
            print(f"일반 모드: 답변용 {self.answer_reserve_tokens} 토큰 예약")

        # 전체 모델 길이에서 답변 예약 토큰을 뺀 입력 가능 길이
        self.model_max_length = 32768
        self.max_input_tokens = self.model_max_length - self.answer_reserve_tokens

        # 청크 1개 ≈ 256토큰, 포맷팅 포함 ≈ 270토큰으로 계산
        self.chunk_estimated_tokens = 270  # 청크 1개당 예상 토큰 수 (256 + 제목/포맷팅 약 10-15토큰)
        # 기본 프롬프트 토큰에 따른 top_k 조정 임계값
        self.length_thresholds = [
            self.max_input_tokens - (2 * self.chunk_estimated_tokens),  # top_k=2
            self.max_input_tokens - (3 * self.chunk_estimated_tokens),  # top_k=3
            self.max_input_tokens - (4 * self.chunk_estimated_tokens),  # top_k=4
            self.max_input_tokens - (5 * self.chunk_estimated_tokens),  # top_k=5
        ]

        self.inp = []
        self.label = []
        self.original_data = []

        # 통계 수집용
        self.topk_usage_stats = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
        self.skipped_count = 0

        mapping = {
                    "category": "카테고리",
                    "topic_keyword": "키워드",
                    "domain": "도메인",
                    "question_type": "질문 유형",
                }

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat_with_dynamic_topk(inp, retrieved_contexts=None):
            """동적 top_k 조정을 포함한 채팅 생성 함수"""
            # question type에 따른 instruction 선택
            instruction = TYPE_INSTRUCTIONS.get(inp['question_type'], "")

            # 기타 정보 생성 (question 제외)
            other_info = {k: v for k, v in inp.items() if k not in ['question']}

            # 기본 chat_parts 구성
            chat_parts = [instruction]

            if other_info:
                chat_parts.append("[기타 정보]")
                info_list = []
                for key, value in other_info.items():
                    if value is not None and value != "":
                        info_list.append(f"{mapping[key]}: {value}")
                chat_parts.append(",".join(info_list))

            # 질문 추가
            chat_parts.append(f"[질문] {inp['question']}")

            # 기본 프롬프트로 토큰 길이 측정
            base_chat = " ".join(chat_parts)
            base_tokens = len(self.tokenizer.encode(base_chat))

            # RAG 컨텍스트 처리
            if retrieved_contexts and len(retrieved_contexts) > 0:
                # 1단계: 기본 필터링 (기존과 동일)
                seen_texts = set()
                unique_contexts = []
                for ctx in retrieved_contexts:
                    text = ctx.get('text', '')
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        unique_contexts.append(ctx)

                # 2단계: 길이 및 반복 필터링
                filtered_contexts = []
                for ctx in unique_contexts:
                    text = ctx.get('text', '')
                    text_no_spaces = text.replace(" ", "")

                    if len(text_no_spaces) <= self.min_context_length:
                        continue

                    if has_excessive_repetition(text, self.max_word_repetition):
                        continue

                    filtered_contexts.append(ctx)

                # 3단계: title별로 그룹화
                title_best_contexts = {}
                for ctx in filtered_contexts:
                    title = ctx.get('title', '')
                    score = ctx.get('score', 0.0)

                    if title not in title_best_contexts or score > title_best_contexts[title].get('score', 0.0):
                        title_best_contexts[title] = ctx

                # 4단계: 동적 top_k 결정 (청크 크기 기반)
                sorted_contexts = sorted(title_best_contexts.values(), key=lambda x: x.get('score', 0.0), reverse=True)

                # 기본 토큰 길이에 따라 top_k 조정
                available_tokens = self.max_input_tokens - base_tokens
                max_possible_contexts = available_tokens // self.chunk_estimated_tokens

                if base_tokens > self.length_thresholds[0]:  # top_k=2
                    current_top_k = min(2, max_possible_contexts)
                elif base_tokens > self.length_thresholds[1]:  # top_k=3
                    current_top_k = min(3, max_possible_contexts)
                elif base_tokens > self.length_thresholds[2]:  # top_k=4
                    current_top_k = min(4, max_possible_contexts)
                else:
                    current_top_k = min(self.top_k, max_possible_contexts)  # 5

                # 최소 1개는 보장하되, 토큰이 부족하면 0개
                current_top_k = max(0, current_top_k)

                # 모든 원본 검색 결과에서 제목 수집 (중복 제거) - 먼저 준비
                all_titles = set()
                for ctx in retrieved_contexts:
                    title = ctx.get('title', '')
                    if title and title != '':
                        all_titles.add(title)

                # 제목 섹션을 고정으로 추가 (모든 검색 결과 제목 포함)
                temp_chat_parts = chat_parts.copy()
                if all_titles:
                    titles_text = "[참고 문서 제목] " + ", ".join(sorted(all_titles))
                    temp_chat_parts.insert(-1, titles_text)  # 질문 앞에 삽입

                # 제목 포함된 기본 토큰 길이 측정
                base_with_titles = " ".join(temp_chat_parts)
                base_with_titles_tokens = len(self.tokenizer.encode(base_with_titles))

                # 컨텍스트를 하나씩 추가하면서 토큰 길이 확인
                final_contexts = []
                for i, ctx in enumerate(sorted_contexts[:current_top_k]):
                    # 임시로 컨텍스트 추가해보기
                    temp_contexts = final_contexts + [ctx]

                    # 참고 문서 섹션만 생성 (제목은 이미 포함됨)
                    context_text = "[참고 문서] "
                    for j, temp_ctx in enumerate(temp_contexts, 1):
                        title = temp_ctx.get('title', '')
                        text = temp_ctx.get('text', '')
                        context_text += f"제목: {title}, 내용: {text} "

                    # 컨텍스트만 토큰화해서 기본 토큰에 더하기 (효율적)
                    context_tokens = len(self.tokenizer.encode(context_text))
                    total_tokens = base_with_titles_tokens + context_tokens

                    if total_tokens <= self.max_input_tokens:
                        final_contexts = temp_contexts
                        actual_top_k = len(final_contexts)
                    else:
                        break  # 토큰 한계 초과하면 중단

                # 최종 컨텍스트가 있으면 chat_parts에 추가
                if final_contexts:
                    # 모든 검색 결과 제목 추가 (이미 수집됨)
                    if all_titles:
                        titles_text = "[참고 문서 제목] " + ", ".join(sorted(all_titles))
                        chat_parts.insert(-1, titles_text)  # 질문 앞에 삽입

                    # 최종 선택된 컨텍스트로 참고 문서 생성
                    context_text = "[참고 문서] "
                    for i, ctx in enumerate(final_contexts, 1):
                        title = ctx.get('title', '')
                        text = ctx.get('text', '')
                        context_text += f"제목: {title}, 내용: {text} "
                    chat_parts.insert(-1, context_text)  # 질문 앞에 삽입

                    actual_top_k = len(final_contexts)
                else:
                    # 컨텍스트는 없지만 제목은 있는 경우
                    if all_titles:
                        titles_text = "[참고 문서 제목] " + ", ".join(sorted(all_titles))
                        chat_parts.insert(-1, titles_text)  # 질문 앞에 삽입
                    actual_top_k = 0

            else:
                actual_top_k = 0

            # 최종 프롬프트 생성
            final_chat = " ".join(chat_parts)
            return final_chat, actual_top_k

        # 데이터 필터링 및 통계 수집
        original_count = len(data)
        filtered_count = 0
        question_filtered = 0
        answer_filtered = 0

        for example in tqdm(data, desc="Loading dataset", unit="example"):
            # 길이 필터링이 활성화된 경우에만 필터링 적용
            if self.enable_length_filtering:
                # 필터링 조건 확인
                question = example["input"]["question"]
                question_type = example["input"].get("question_type", "")

                # 조건 1: question이 공백 제거 후 500자 초과인 경우 제외
                if len(question) > self.remove_question_count:
                    question_filtered += 1
                    continue

                # 조건 2: 서술형이면서 answer가 공백 제거 후 550자 초과인 경우 제외
                if question_type == "서술형" and "output" in example:
                    answer = example["output"].get("answer", "")
                    if len(answer) > self.remove_answer_count:
                        answer_filtered += 1
                        continue

            # 필터링 통과한 데이터만 처리
            self.original_data.append(example)

            # 미리 검색된 RAG 결과 사용
            retrieved_contexts = None
            if self.use_rag and "retrieved_contexts" in example:
                retrieved_contexts = example["retrieved_contexts"]

            # 동적 top_k로 프롬프트 생성
            user_prompt, used_top_k = make_chat_with_dynamic_topk(example["input"], retrieved_contexts)

            # 통계 수집
            self.topk_usage_stats[used_top_k] += 1

            # use_system_prompt 설정에 따라 메시지 구성 변경
            if self.use_system_prompt:
                # 시스템 프롬프트를 별도 역할로 사용
                message = [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": user_prompt},
                ]
            else:
                # 시스템 프롬프트를 사용자 메시지 앞에 추가
                if self.prompt:
                    combined_prompt = f"{self.prompt} {user_prompt}"
                else:
                    combined_prompt = user_prompt

                message = [
                    {"role": "user", "content": combined_prompt},
                ]

            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )

            # 최종 입력 토큰 길이 체크
            if source[0].shape[0] > self.max_input_tokens:
                print(f"Skipping sample: input token length {source[0].shape[0]} exceeds limit {self.max_input_tokens}")
                self.skipped_count += 1
                continue

            target = example.get("output", "")
            target_text = ""

            if target != "":
                target_text = target.get("answer", "")
                target_text += tokenizer.eos_token

            target = tokenizer(
                target_text,
                return_attention_mask=False,
                add_special_tokens=False,
                return_tensors="pt"
            )
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([self.igonore_index] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

            filtered_count += 1

        # 필터링 결과 출력
        print(f"\n=== 데이터 필터링 결과 ===")
        print(f"모델 최대 길이: {self.model_max_length}")
        print(f"답변 예약 토큰: {self.answer_reserve_tokens} ({'CoT' if self.use_cot else '일반'})")
        print(f"입력 최대 토큰: {self.max_input_tokens}")
        print(f"길이 필터링 활성화: {'예' if self.enable_length_filtering else '아니오'}")
        print(f"원본 데이터 수: {original_count}")
        if self.enable_length_filtering:
            print(f"질문 길이 초과로 제외된 데이터: {question_filtered} (질문 > {self.remove_question_count}자)")
            print(f"서술형 답변 길이 초과로 제외된 데이터: {answer_filtered} (서술형 답변 > {self.remove_answer_count}자)")
        print(f"토큰 길이 초과로 건너뛴 데이터: {self.skipped_count}")
        print(f"최종 사용된 데이터 수: {filtered_count}")
        print(f"단어 반복 필터링: 동일 단어 {self.max_word_repetition}회 이상 반복 시 제거")
        if self.enable_length_filtering:
            print(f"제외 비율: {((original_count - filtered_count) / original_count * 100):.2f}%")

        # Top-k 사용 통계 출력
        print(f"\n=== Top-k 사용 통계 (청크 256토큰 기준) ===")
        print(f"청크당 예상 토큰: {self.chunk_estimated_tokens} (256 + 제목/포맷팅)")
        print(f"동적 조정 임계값: {self.length_thresholds}")
        total_with_contexts = sum(v for k, v in self.topk_usage_stats.items() if k > 0)
        for k, count in sorted(self.topk_usage_stats.items(), reverse=True):
            if k == 0:
                print(f"컨텍스트 없음: {count}개")
            else:
                percentage = (count / total_with_contexts * 100) if total_with_contexts > 0 else 0
                estimated_tokens = k * self.chunk_estimated_tokens
                print(f"Top-{k} 사용: {count}개 ({percentage:.1f}%) - 약 {estimated_tokens}토큰")

    def check_token_lengths(self):
        """토큰 길이 통계 확인"""
        if not self.inp:
            print("데이터가 없습니다.")
            return

        lengths = [len(inp) for inp in self.inp]

        print(f"\n=== 토큰 길이 통계 ===")
        print(f"총 데이터 수: {len(lengths)}")
        print(f"최소 길이: {min(lengths)}")
        print(f"최대 길이: {max(lengths)}")
        print(f"평균 길이: {sum(lengths)/len(lengths):.1f}")
        print(f"중간값: {sorted(lengths)[len(lengths)//2]}")

        # 길이별 분포
        over_30k = sum(1 for l in lengths if l > 30000)
        over_32k = sum(1 for l in lengths if l > 32768)
        over_max = sum(1 for l in lengths if l > self.max_input_tokens)

        print(f"\n=== 길이 초과 통계 ===")
        print(f"30,000 토큰 초과: {over_30k}/{len(lengths)} ({over_30k/len(lengths)*100:.1f}%)")
        print(f"32,768 토큰 초과: {over_32k}/{len(lengths)} ({over_32k/len(lengths)*100:.1f}%)")
        print(f"입력 한계({self.max_input_tokens}) 초과: {over_max}/{len(lengths)} ({over_max/len(lengths)*100:.1f}%)")

        # 길이 구간별 분포
        print(f"\n=== 길이 구간별 분포 ===")
        ranges = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000), (40000, float('inf'))]
        for start, end in ranges:
            count = sum(1 for l in lengths if start <= l < end)
            if end == float('inf'):
                print(f"{start}+ 토큰: {count}개 ({count/len(lengths)*100:.1f}%)")
            else:
                print(f"{start}-{end} 토큰: {count}개 ({count/len(lengths)*100:.1f}%)")

        return {
            'lengths': lengths,
            'min': min(lengths),
            'max': max(lengths),
            'mean': sum(lengths)/len(lengths),
            'over_limit': over_max
        }

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inp[idx],
            "labels":   self.label[idx],
            "original_data": self.original_data[idx]
        }

class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
