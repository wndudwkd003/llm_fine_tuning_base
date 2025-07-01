import os
import json
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import LoraConfig
from src.configs.config import (
    SystemArgs,
    ModelArgs,
    DataArgs,
    TrainingArgs,
    BitsAndBytesArgs,
    LoraArgs
)
from src.utils.path_utils import create_out_dir # , output_path_record
from src.utils.checker import check_sft_type


def initialize_config(
    system_args: SystemArgs,
    bnb_args: BitsAndBytesArgs,
    lora_args: LoraArgs,
    training_args: TrainingArgs
):
    bnb_config = None
    lora_config = None

    if system_args.use_qlora:
        bnb_config = BitsAndBytesConfig(**vars(bnb_args))
    if system_args.use_lora:
        lora_config = LoraConfig(**vars(lora_args))

    training_config = TrainingArguments(**vars(training_args))
    return bnb_config, lora_config, training_config


def add_system_prompt(example, system_prompt: str):
    example['messages'] = [{"role": "system", "content": system_prompt}] + example['messages']
    return example


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt_ids,
    terminators,
    model_args: ModelArgs,
    device="cuda",
):
    attention_mask = torch.ones_like(prompt_ids)
    outputs = model.generate(
        input_ids=prompt_ids.to(device).unsqueeze(0),
        attention_mask=attention_mask.to(device).unsqueeze(0),
        do_sample=model_args.do_sample,
        max_new_tokens=model_args.max_new_tokens,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        temperature=model_args.temperature,
        top_p=model_args.top_p,
        top_k=model_args.top_k,
        repetition_penalty=model_args.repetition_penalty,
    )
    gen_tokens = outputs[0][prompt_ids.size(0):]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    if text.startswith("[|assistant|]"):
        text = text[len("[|assistant|]"):].lstrip()
    if text.startswith("assistant\n\n"):
        text = text[len("assistant\n\n"):]
    if text.startswith("답변: "):
        text = text[4:]
    elif text.startswith("답변:"):
        text = text[3:]
    if "#" in text:
        text = text.split("#", 1)[0].strip()
    return text


def run_inference(
        test_dataset,
        model_dir: str,
        model_args: ModelArgs,
        bnb_config: BitsAndBytesConfig
):
    print("\n--- [INFO] Starting Inference ---")

    # 1) 모델 및 토크나이저 로드 (기존과 동일)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        attn_implementation="flash_attention_2" if model_args.use_flash_attn2 else "eager",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=model_args.dtype.value,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) 답변 생성
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]

    results = []
    for example in tqdm(test_dataset, desc="Inference"):
        prompt_ids = tokenizer.apply_chat_template(
            conversation=example['messages'],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).input_ids

        answer_text = generate_answer(
            model,
            tokenizer,
            prompt_ids[0],
            terminators,
            model_args=model_args,
        )

        new_example = example.copy()
        new_example['output'] = {'answer': answer_text}
        results.append(new_example)

    # 3) 파일 저장
    save_path = os.path.join(model_dir, "predictions.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Inference finished. Predictions saved to: {save_path}")


def main(
    system_args: SystemArgs,
    model_args: ModelArgs,
    data_args: DataArgs,
    training_args: TrainingArgs,
    bits_and_bytes_args: BitsAndBytesArgs,
    lora_args: LoraArgs
):

    # 0) 저장 경로 및 백업 설정
    output_dir = create_out_dir(
        training_args.output_dir,
        check_sft_type(system_args.use_lora, system_args.use_qlora),
        model_args.model_id.value,
        system_args.additional_info,
        backup_path=system_args.backup_path
    )
    training_args.output_dir = output_dir

    # 1) 설정 파일 업데이트
    bnb_config, lora_config, training_config = initialize_config(
        system_args,
        bits_and_bytes_args,
        lora_args,
        training_args
    )

    # 2) 데이터 세트 로드
    data_splits = []
    if system_args.train:
        data_splits.extend(["train", "dev"])
    if system_args.test:
        data_splits.append("test")

    data_files = {sp: os.path.join(data_args.data_dir, f"{sp}.json") for sp in data_splits}
    data_dict = load_dataset("json",data_files=data_files, num_proc=system_args.num_proc)
    data_dict = data_dict.map(
        lambda example: add_system_prompt(example, system_prompt=model_args.prompt_template),
        num_proc=system_args.num_proc
    )


    if system_args.train:
        print("[INFO] Starting Training")

        # 3) 모델, 토크나이저
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_id.value,
            trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token


        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_id.value,
            attn_implementation="flash_attention_2" if model_args.use_flash_attn2 else "eager",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=model_args.dtype.value,
        )

        model.config.use_cache = False

        # 4) Trainer 설정
        trainer = SFTTrainer(
            model=model,
            args=training_config,
            train_dataset=data_dict["train"],
            eval_dataset=data_dict["dev"],
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        # 5) 모델 학습
        trainer.train()

        # 6) 모델 저장
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(training_config.output_dir)
        tokenizer.save_pretrained(training_config.output_dir)
        print(f"--- [INFO] Merged model saved to {training_config.output_dir}")

    # 5) 추론 (system_args.test가 True일 때 실행)
    if system_args.test:
        run_inference(
            test_dataset=data_dict["test"],
            model_dir=training_config.output_dir,
            model_args=model_args,
            bnb_config=bnb_config,
        )


if __name__ == "__main__":
    system_args = SystemArgs()
    model_args = ModelArgs()
    data_args = DataArgs()
    training_args = TrainingArgs()
    bits_and_bytes_args = BitsAndBytesArgs()
    lora_args = LoraArgs()

    os.environ["HF_TOKEN"] = system_args.hf_token
    os.environ["CUDA_VISIBLE_DEVICES"] = str(system_args.gpu_number)

    main(system_args, model_args, data_args, training_args, bits_and_bytes_args, lora_args)
