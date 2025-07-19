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
                    "[지시사항] 질문을 잘 읽고 2단어 내외의 단답형으로 답하시오. 문제를 그대로 출력하지 마시오."

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
    ):
        self.igonore_index = igonore_index
        self.prompt = prompt
        self.use_system_prompt = use_system_prompt
        self.use_rag = use_rag
        self.is_test_and_drop_other_info = is_test_and_drop_other_info
        self.enable_length_filtering = enable_length_filtering

        self.remove_question_count = 400
        self.remove_answer_count = 500
        self.top_k = 5
        self.min_context_length = 15  # 최소 컨텍스트 길이
        self.max_word_repetition = 5  # 최대 단어 반복 허용 횟수

        self.inp = []
        self.label = []
        self.original_data = []

        mapping = {
                    "category": "카테고리",
                    "topic_keyword": "키워드",
                    "domain": "도메인",
                    "question_type": "질문 유형",
                }

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp, retrieved_contexts=None):
            # question type에 따른 instruction 선택
            instruction = TYPE_INSTRUCTIONS.get(inp['question_type'], "")

            # 기타 정보 생성 (question 제외)
            other_info = {k: v for k, v in inp.items() if k not in ['question']}

            # 기타 정보가 있는 경우에만 추가
            chat_parts = [instruction]

            # RAG로 검색한 문서가 있으면 추가
            if retrieved_contexts and len(retrieved_contexts) > 0:
                # 1단계: 텍스트 중복 제거 (전체 컨텍스트에 대해)
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

                    # 길이 체크
                    if len(text_no_spaces) <= self.min_context_length:
                        continue

                    # 단어 반복 체크
                    if has_excessive_repetition(text, self.max_word_repetition):
                        continue

                    filtered_contexts.append(ctx)

                # 3단계: title별로 그룹화하고 각 title에서 가장 높은 score를 가진 컨텍스트만 선택
                title_best_contexts = {}
                for ctx in filtered_contexts:
                    title = ctx.get('title', '')
                    score = ctx.get('score', 0.0)

                    # 같은 title이 없거나, 더 높은 score를 가진 경우에만 저장
                    if title not in title_best_contexts or score > title_best_contexts[title].get('score', 0.0):
                        title_best_contexts[title] = ctx

                # 4단계: score 순으로 정렬하여 상위 5개 선택
                final_contexts = sorted(title_best_contexts.values(), key=lambda x: x.get('score', 0.0), reverse=True)[:self.top_k]

                # 5단계: 검색 제목 및 참고 문서 생성
                if final_contexts:
                    # 모든 원본 검색 결과에서 제목 수집 (중복 제거)
                    all_titles = set()
                    for ctx in retrieved_contexts:
                        title = ctx.get('title', '')
                        if title and title != '':
                            all_titles.add(title)

                    # 검색 제목 섹션 추가
                    if all_titles:
                        titles_text = "[참고 문서 제목] " + ", ".join(sorted(all_titles))
                        chat_parts.append(titles_text)

                    # 최종 선택된 컨텍스트로 참고 문서 생성
                    context_text = "[참고 문서] "
                    for i, ctx in enumerate(final_contexts, 1):
                        title = ctx.get('title', '')
                        text = ctx.get('text', '')
                        context_text += f"제목: {title}, 내용: {text} "
                    chat_parts.append(context_text)

            if other_info:
                chat_parts.append("[기타 정보]")
                info_list = []
                for key, value in other_info.items():
                    if value is not None and value != "":
                        info_list.append(f"{mapping[key]}: {value}")
                chat_parts.append(",".join(info_list))

            # 질문 추가
            chat_parts.append(f"[질문] {inp['question']}")

            # 최종 프롬프트 생성
            chat = " ".join(chat_parts)

            return chat

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

            user_prompt = make_chat(example["input"], retrieved_contexts)

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
        print(f"길이 필터링 활성화: {'예' if self.enable_length_filtering else '아니오'}")
        print(f"원본 데이터 수: {original_count}")
        if self.enable_length_filtering:
            print(f"질문 길이 초과로 제외된 데이터: {question_filtered} (질문 > {self.remove_question_count}자)")
            print(f"서술형 답변 길이 초과로 제외된 데이터: {answer_filtered} (서술형 답변 > {self.remove_answer_count}자)")
        print(f"최종 사용된 데이터 수: {filtered_count}")
        print(f"단어 반복 필터링: 동일 단어 {self.max_word_repetition}회 이상 반복 시 제거")
        if self.enable_length_filtering:
            print(f"제외 비율: {((original_count - filtered_count) / original_count * 100):.2f}%")

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
