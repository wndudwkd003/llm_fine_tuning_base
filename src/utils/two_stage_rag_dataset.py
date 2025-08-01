import json
import os
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
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

# 1단계: 검색 질의 생성용 지시사항
QUERY_GENERATION_INSTRUCTION = """다음 질문에 대한 답변을 위해 필요한 검색 질의 3개를 생성하세요:

1) 핵심 키워드 기반 검색 질의 - 답변에 필요한 핵심 개념이나 용어를 포함한 질의
2) 맥락 정보 검색 질의 - 문제의 배경지식이나 관련 정보를 찾기 위한 질의
3) 상세 정보 검색 질의 - 답변을 뒷받침하는 구체적인 사실이나 데이터를 찾기 위한 질의

각 질의는 <검색></검색> 태그로 감싸서 출력하세요."""

class TwoStageRAGDataset(Dataset):
    def __init__(
            self,
            fname,
            tokenizer,
            stage="query_generation",  # "query_generation" or "answer_generation"
            ignore_index=-100,
            prompt="",
            use_system_prompt=False,
            is_test_and_drop_other_info=False
    ):
        self.ignore_index = ignore_index
        self.prompt = prompt
        self.use_system_prompt = use_system_prompt
        self.stage = stage
        self.is_test_and_drop_other_info = is_test_and_drop_other_info

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
            data = data[:int(len(data) * 0.01)]

        def make_query_generation_chat(inp):
            """1단계: 검색 질의 생성용 프롬프트"""
            instruction = QUERY_GENERATION_INSTRUCTION

            # 기타 정보 생성 (question과 question_type 제외)
            other_info = {k: v for k, v in inp.items() if k not in ['question', 'question_type']}

            chat_parts = [instruction]

            if other_info:
                chat_parts.append("[기타 정보]")
                info_list = []
                for key, value in other_info.items():
                    info_list.append(f"{mapping[key]}: {value}")
                chat_parts.append(",".join(info_list))

            # 질문 추가
            chat_parts.append(f"[질문] {inp['question']}")

            return " ".join(chat_parts)

        def make_answer_generation_chat(inp, retrieved_contexts=None):
            """2단계: 최종 답변 생성용 프롬프트"""
            # question type에 따른 instruction 선택
            instruction = TYPE_INSTRUCTIONS.get(inp['question_type'], "")

            # 기타 정보 생성 (question과 question_type 제외)
            other_info = {k: v for k, v in inp.items() if k not in ['question', 'question_type']}

            chat_parts = [instruction]

            # RAG로 검색한 문서가 있으면 추가
            if retrieved_contexts and len(retrieved_contexts) > 0:
                context_text = "[참고 문서]\n"
                for i, ctx in enumerate(retrieved_contexts, 1):
                    context_text += f"{i}. {ctx['text']}\n"
                chat_parts.append(context_text)

            if other_info:
                chat_parts.append("[기타 정보]")
                info_list = []
                for key, value in other_info.items():
                    info_list.append(f"{mapping[key]}: {value}")
                chat_parts.append(",".join(info_list))

            # 질문 추가
            chat_parts.append(f"[질문] {inp['question']}")

            return " ".join(chat_parts)

        for example in tqdm(data, desc=f"Loading {stage} dataset", unit="example"):
            self.original_data.append(example)

            if self.stage == "query_generation":
                # 1단계: 검색 질의 생성
                user_prompt = make_query_generation_chat(example["input"])

                # 목표 출력: raw_response (검색 태그가 포함된 질의들)
                target_text = example["output"]["raw_response"]

            elif self.stage == "answer_generation":
                # 2단계: 최종 답변 생성
                retrieved_contexts = example.get("retrieved_contexts", [])
                user_prompt = make_answer_generation_chat(example["input"], retrieved_contexts)

                # 목표 출력: 최종 답변
                target_text = example["output"]["answer"]

            else:
                raise ValueError(f"Invalid stage: {self.stage}")

            # use_system_prompt 설정에 따라 메시지 구성 변경
            if self.use_system_prompt:
                message = [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": user_prompt},
                ]
            else:
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

            # target_text += tokenizer.eos_token
            target_text = re.sub(r'\s+', ' ', target_text)  # 모든 공백문자를 하나의 스페이스로
            target_text = target_text.strip()  # 앞뒤 공백 제거
            target_text += tokenizer.eos_token
            print("Target Text:", target_text)

            target = tokenizer(
                target_text,
                return_attention_mask=False,
                add_special_tokens=False,
                return_tensors="pt"
            )
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([self.ignore_index] * source[0].shape[0]), target["input_ids"][0]))

            self.inp.append(input_ids)
            self.label.append(labels)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inp[idx],
            "labels": self.label[idx],
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

# 사용 예시
def create_datasets(data_file, tokenizer):
    """두 단계 모두에 대한 데이터셋 생성"""

    # 1단계: 검색 질의 생성 데이터셋
    query_gen_dataset = TwoStageRAGDataset(
        fname=data_file,
        tokenizer=tokenizer,
        stage="query_generation"
    )

    # 2단계: 최종 답변 생성 데이터셋
    answer_gen_dataset = TwoStageRAGDataset(
        fname=data_file,
        tokenizer=tokenizer,
        stage="answer_generation"
    )

    return query_gen_dataset, answer_gen_dataset

# 추론 시 사용할 함수들
def extract_queries_from_response(response_text):
    """모델 응답에서 검색 질의 추출"""
    import re
    queries = re.findall(r'<검색>(.*?)</검색>', response_text)
    return queries

def deduplicate_and_rank_contexts(contexts, max_contexts=4):
    """중복 제거 및 점수 기준 상위 선택"""
    # 텍스트 기준으로 중복 제거
    seen_texts = set()
    unique_contexts = []

    for ctx in contexts:
        if ctx['text'] not in seen_texts:
            seen_texts.add(ctx['text'])
            unique_contexts.append(ctx)

    # 점수 기준 정렬 (내림차순)
    unique_contexts.sort(key=lambda x: x['score'], reverse=True)

    # 최대 개수만큼 선택
    return unique_contexts[:max_contexts]
