import json
import os
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


TYPE_INSTRUCTIONS = {
                "선다형": (
                    "[지시사항] 질문을 잘 읽고 주어진 보기 중에서 정답을 숫자로만 답변하시오. 문제를 그대로 출력하지 마시오."
                ),
                "서술형": (
                    "[지시사항] 질문을 잘 읽고 300자~500자 이내의 자연스러운 문법으로 완성된 서술형으로 답변하세요. 최대한 자세히 적되, 핵심 단어를 놓치지 말고 정확하게 사실만 답하여야합니다. 그리고 문제를 그대로 출력하지 마시오."
                ),
                "단답형": (
                    "[지시사항] 질문을 잘 읽고 5어절 이하의 단답형으로 답하시오. 문제를 그대로 출력하지 마시오."

                ),
            }

class CustomDataset(Dataset):
    def __init__(
            self,
            fname,
            tokenizer,
            use_rag=False,   # RAG 사용 여부
            igonore_index=-100,
            prompt="",
            use_system_prompt=False,
            is_test_and_drop_other_info=False
    ):
        self.igonore_index = igonore_index
        self.prompt = prompt
        self.use_system_prompt = use_system_prompt
        self.use_rag = use_rag
        self.is_test_and_drop_other_info = is_test_and_drop_other_info

        self.remove_question_count = 500
        self.remove_answer_count = 500

        self.inp = []
        self.label = []
        self.original_data = []

        mapping = {
                    "category": "카테고리",
                    "topic_keyword": "키워드",
                    "domain": "도메인",
                }

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp, retrieved_contexts=None):
            # question type에 따른 instruction 선택
            instruction = TYPE_INSTRUCTIONS.get(inp['question_type'], "")

            # 기타 정보 생성 (question과 question_type 제외)
            other_info = {k: v for k, v in inp.items() if k not in ['question', 'question_type']}

            # 테스트 모드이고 단답형인 경우 topic_keyword 제거
            # if self.is_test_and_drop_other_info and inp.get('question_type') == '단답형':
            #     # other_info.pop('topic_keyword', None)
            #     other_info["topic_keyword"] = ""  # topic_keyword를 빈 문자열로 설정

            # 기타 정보가 있는 경우에만 추가
            chat_parts = [instruction]


            # RAG로 검색한 문서가 있으면 추가
            if retrieved_contexts and len(retrieved_contexts) > 0:
                # 1단계: 텍스트 길이 필터링 (15자 이하 제거)
                filtered_contexts = []
                for ctx in retrieved_contexts:
                    text = ctx.get('text', '')
                    # 공백 제거 후 실제 텍스트 길이 확인
                    cleaned_text = text.strip()
                    if len(cleaned_text) > 15:
                        filtered_contexts.append(ctx)

                # 2단계: title별로 그룹화하여 최고 점수만 유지
                title_best_contexts = {}
                for ctx in filtered_contexts:
                    title = ctx.get('title', '제목 없음')
                    score = ctx.get('score', 0.0)

                    # 해당 title이 처음 나오거나, 더 높은 점수인 경우 업데이트
                    if title not in title_best_contexts or score > title_best_contexts[title].get('score', 0.0):
                        title_best_contexts[title] = ctx

                # 3단계: title별 최고 점수 컨텍스트들을 점수 기준으로 정렬 후 상위 선택
                unique_contexts = list(title_best_contexts.values())
                unique_contexts = sorted(unique_contexts, key=lambda x: x.get('score', 0.0), reverse=True)[:5]

                # 선택된 컨텍스트로 참고 문서 생성
                if unique_contexts:  # 필터링 후에도 컨텍스트가 있는 경우에만
                    context_text = "[참고 문서] "
                    for i, ctx in enumerate(unique_contexts, 1):
                        title = ctx.get('title', '제목 없음')
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
            # 필터링 조건 확인
            question = example["input"]["question"]
            question_type = example["input"].get("question_type", "")

            # 조건 1: question이 공백 제거 후 500자 초과인 경우 제외
            # question_no_spaces = question.replace(" ", "")
            if len(question) > self.remove_question_count:
                question_filtered += 1
                continue

            # 조건 2: 서술형이면서 answer가 공백 제거 후 550자 초과인 경우 제외
            if question_type == "서술형" and "output" in example:
                answer = example["output"].get("answer", "")
                # answer_no_spaces = answer.replace(" ", "")
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

            # print("message:", message)

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

            # print("target_text:", target_text)

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
        print(f"원본 데이터 수: {original_count}")
        print(f"질문 길이 초과로 제외된 데이터: {question_filtered} (질문 공백제거 후 > {self.remove_question_count}자)")
        print(f"서술형 답변 길이 초과로 제외된 데이터: {answer_filtered} (서술형 답변 공백제거 후 > {self.remove_answer_count}자)")
        print(f"최종 사용된 데이터 수: {filtered_count}")
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
