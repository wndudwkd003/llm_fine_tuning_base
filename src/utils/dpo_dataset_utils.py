import json
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.configs.config import ModelArgs, SFTTrainingArgs
from src.utils.print_utils import printi
from src.utils.model_utils import generate_answer


def create_dpo_dataset(
    source_dataset,
    model,
    tokenizer,
    model_args: ModelArgs,
    sft_training_args: SFTTrainingArgs,
    output_path: str
):
    printi("Starting DPO dataset creation.")

    # 1) EOT 토큰 설정
    eot_token_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
    terminators = [tokenizer.eos_token_id] if eot_token_id is None else [tokenizer.eos_token_id, eot_token_id]

    dpo_results = []
    for example in tqdm(source_dataset, desc="Generating rejected responses"):
        # 2) 프롬프트와 정답(chosen) 분리
        prompt_messages = [msg for msg in example['messages'] if msg['role'] == 'user']
        chosen_response = next((msg['content'] for msg in reversed(example['messages']) if msg['role'] == 'assistant'), None)

        if not prompt_messages or chosen_response is None:
            continue

        prompt_text = prompt_messages[-1]['content']

        # 3) 모델 추론을 위한 입력값 생성
        prompt_ids = tokenizer.apply_chat_template(
            conversation=[{'role': 'user', 'content': prompt_text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # 4) 모델을 통해 'rejected' 답변 생성
        rejected_response = generate_answer(
            model,
            tokenizer,
            prompt_ids[0],
            terminators,
            model_args=model_args,
        )

        # ==================== 출력문 추가 ====================
        print("\n" + "="*10)
        print(f"[프롬프트 (질문)]\n{prompt_text}")
        print("-" * 80)
        print(f"[정답 (Chosen)]\n{chosen_response}")
        print("-" * 80)
        print(f"[생성된 답변 (Rejected)]\n{rejected_response}")
        print("="*10)
        # ====================================================

        # 5) DPO 데이터 형식으로 저장
        dpo_results.append({
            "prompt": prompt_text,
            "chosen": chosen_response,
            "rejected": rejected_response
        })

    # 6) 파일로 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dpo_results, f, ensure_ascii=False, indent=4)

    printi(f"DPO dataset created and saved to: {output_path}")
