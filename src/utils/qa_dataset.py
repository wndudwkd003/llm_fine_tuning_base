import json
import os
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# question type별 instruction 정의
TYPE_INSTRUCTIONS = {
    "선다형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침] 주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.\n\n"
    ),
    "서술형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침] 질문에 대한 답변을 완성된 문장으로 서술하시오.\n\n"
    ),
    "단답형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침] 질문에 대한 답을 단답형으로 답하시오. 예시) [질문] 뉴턴과 과련된 나무열매는? [답변] 사과\n\n"
    ),
    "교정형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침] 주어진 문장이 올바른지 판단하고, 틀린 경우 올바르게 교정하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
    ),
    "선택형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침] 주어진 보기들 중에서 가장 적절한 것을 선택하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
    )
}

class CustomDataset(Dataset):
    def __init__(
            self,
            fname,
            tokenizer,
            use_rag=False,   # RAG 사용 여부
            igonore_index=-100,
            prompt="",
            use_system_prompt=False
    ):
        self.igonore_index = igonore_index
        self.prompt = prompt
        self.use_system_prompt = use_system_prompt
        self.use_rag = use_rag

        self.inp = []
        self.label = []
        self.original_data = []

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp, retrieved_contexts=None):
            # question type에 따른 instruction 선택
            instruction = TYPE_INSTRUCTIONS.get(inp['question_type'], "")

            # 기타 정보 생성 (question과 question_type 제외)
            other_info = {k: v for k, v in inp.items() if k not in ['question', 'question_type']}

            # 기타 정보가 있는 경우에만 추가
            chat_parts = [instruction]

            # RAG로 검색한 문서가 있으면 추가
            if retrieved_contexts and len(retrieved_contexts) > 0:
                context_text = "[참고 문서]\n"
                for i, ctx in enumerate(retrieved_contexts, 1):
                    context_text += f"{i}. {ctx['text']}\n"
                chat_parts.append(context_text)

            if other_info:
                info_list = ["[기타 정보]"]
                for key, value in other_info.items():
                    info_list.append(f"- {key}: {value}")
                chat_parts.append(" ".join(info_list))

            # 질문 추가
            chat_parts.append(f"[질문] {inp['question']}")

            # 최종 프롬프트 생성
            chat = "\n\n".join(chat_parts)

            return chat

        for example in tqdm(data, desc="Loading dataset", unit="example"):
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
                    combined_prompt = f"{self.prompt}\n\n{user_prompt}"
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

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {
            "id":       self.original_data[idx]["id"],
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
