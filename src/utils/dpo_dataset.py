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
                    "[지침] 질문에 대한 답을 단답형으로 답하시오.\n\n"
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

class DPODataset(Dataset):
    def __init__(
            self,
            fname,
            tokenizer,
            prompt="",
            use_system_prompt=False,
    ):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.use_system_prompt = use_system_prompt

        self.data = []
        self.original_data = []

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            # question type에 따른 instruction 선택
            instruction = TYPE_INSTRUCTIONS.get(inp['question_type'], "")

            # 기타 정보 생성 (question과 question_type 제외)
            other_info = {k: v for k, v in inp.items() if k not in ['question', 'question_type']}

            # 기타 정보가 있는 경우에만 추가
            chat_parts = [instruction]
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

        for example_idx, example in enumerate(tqdm(data, desc="Loading DPO dataset", unit="example")):
            user_prompt = make_chat(example["input"])

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

            # 프롬프트 토큰화 (공통 부분)
            prompt_tokens = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )

            # DPO 데이터에서 chosen과 rejected 응답 추출
            chosen_responses = []
            rejected_responses = []

            if "output" in example:
                # cot_answer 배열에서 chosen 응답들 추출
                if "cot_answer" in example["output"]:
                    chosen_responses = example["output"]["cot_answer"]

                # reject 배열에서 rejected 응답들 추출
                if "reject" in example["output"]:
                    rejected_responses = example["output"]["reject"]

            # 모든 조합 생성 (4 x 4 = 16개)
            for chosen_idx, chosen_response in enumerate(chosen_responses):
                for rejected_idx, rejected_response in enumerate(rejected_responses):
                    # chosen과 rejected 응답 토큰화
                    chosen_tokens = self.tokenizer(
                        chosen_response,
                        return_attention_mask=False,
                        add_special_tokens=False,
                        return_tensors="pt"
                    )["input_ids"]

                    rejected_tokens = self.tokenizer(
                        rejected_response,
                        return_attention_mask=False,
                        add_special_tokens=False,
                        return_tensors="pt"
                    )["input_ids"]

                    # 전체 시퀀스 구성
                    chosen_input_ids = torch.concat((prompt_tokens[0], chosen_tokens[0]))
                    rejected_input_ids = torch.concat((prompt_tokens[0], rejected_tokens[0]))

                    # 각 조합을 별도 데이터포인트로 추가
                    self.data.append({
                        "prompt": prompt_tokens[0],
                        "chosen": chosen_input_ids,
                        "rejected": rejected_input_ids,
                        "chosen_labels": chosen_tokens[0],
                        "rejected_labels": rejected_tokens[0],
                    })

                    # 원본 데이터에 조합 정보 추가
                    original_with_combo = example.copy()
                    original_with_combo["combo_info"] = {
                        "chosen_idx": chosen_idx,
                        "rejected_idx": rejected_idx,
                        "combo_id": f"{example.get('id', example_idx)}_{chosen_idx}_{rejected_idx}"
                    }
                    self.original_data.append(original_with_combo)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "id": self.original_data[idx].get("combo_info", {}).get("combo_id", idx),
            "prompt": self.data[idx]["prompt"],
            "chosen": self.data[idx]["chosen"],
            "rejected": self.data[idx]["rejected"],
            "chosen_labels": self.data[idx]["chosen_labels"],
            "rejected_labels": self.data[idx]["rejected_labels"],
            "original_data": self.original_data[idx],
            "combo_info": self.original_data[idx].get("combo_info", {})
        }

class DPODataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # 배치의 각 요소들을 분리
        prompts = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]
        chosen_labels = [item["chosen_labels"] for item in batch]
        rejected_labels = [item["rejected_labels"] for item in batch]

        # 패딩 처리
        prompts = torch.nn.utils.rnn.pad_sequence(
            prompts, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        chosen = torch.nn.utils.rnn.pad_sequence(
            chosen, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        rejected = torch.nn.utils.rnn.pad_sequence(
            rejected, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        chosen_labels = torch.nn.utils.rnn.pad_sequence(
            chosen_labels, batch_first=True, padding_value=-100
        )

        rejected_labels = torch.nn.utils.rnn.pad_sequence(
            rejected_labels, batch_first=True, padding_value=-100
        )

        return {
            "prompt_input_ids": prompts,
            "chosen_input_ids": chosen,
            "rejected_input_ids": rejected,
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
            "prompt_attention_mask": prompts.ne(self.tokenizer.pad_token_id),
            "chosen_attention_mask": chosen.ne(self.tokenizer.pad_token_id),
            "rejected_attention_mask": rejected.ne(self.tokenizer.pad_token_id),
        }
