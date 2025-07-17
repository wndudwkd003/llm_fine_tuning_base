import os, json
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import (
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import PeftModel
from src.configs.config import (
    SystemArgs,
    ModelArgs,
    DataArgs,
    DPOTrainingArgs,  # 새로 추가한 DPOTrainingArgs
    BitsAndBytesArgs,
    LoraArgs,
    # FSDPArgs
)

from sklearn.metrics import accuracy_score

from src.utils.seeds import set_seed
from src.utils.path_utils import create_out_dir
from src.utils.checker import check_sft_type
from src.utils.log_utils import save_training_curves
from src.utils.print_utils import printw, printi, printe
from src.utils.model_utils import (
    initialize_config,
    data_prepare,
    generate_answer,
    prepare_model_tokenmizer
)
from src.utils.dpo_dataset import DPODataset  # 새로 만든 DPO dataset import


def run_inference(
        model_dir: str,
        model_args: ModelArgs,
        bnb_config: BitsAndBytesConfig,
        target_name: str,
        dpo_training_args: DPOTrainingArgs,
):
    printi("Starting Inference")

    # 1) 모델 및 토크나이저 로드
    model, tokenizer = prepare_model_tokenmizer(
        model_args=model_args,
        bnb_config=bnb_config,
        is_train=False,
        gradient_checkpointing=dpo_training_args.gradient_checkpointing
    )

    # DPO용 데이터 로드
    test_dataset = DPODataset(
        fname=os.path.join(data_args.data_dir, "test.json"),
        tokenizer=tokenizer,
        prompt=model_args.prompt_template if model_args.use_system_prompt else "",
        use_system_prompt=model_args.use_system_prompt
    )

    # adapter 디렉토리 확인 및 로드
    adapter_dir = os.path.join(model_dir, model_args.load_model)

    # adapter 디렉토리가 존재하는지 확인
    if not os.path.exists(adapter_dir):
        printe(f"Adapter directory not found: {adapter_dir}")
        printe("Available directories:")
        if os.path.exists(model_dir):
            for item in os.listdir(model_dir):
                if os.path.isdir(os.path.join(model_dir, item)):
                    printe(f"  - {item}")
        else:
            printe(f"Model directory does not exist: {model_dir}")
        return

    # adapter_config.json 파일이 존재하는지 확인
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        printe(f"adapter_config.json not found in: {adapter_dir}")
        printe("Available files in adapter directory:")
        if os.path.exists(adapter_dir):
            for item in os.listdir(adapter_dir):
                printe(f"  - {item}")
        return

    try:
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
        )
        printi(f"Successfully loaded adapter from: {adapter_dir}")
    except Exception as e:
        printe(f"Failed to load adapter: {e}")
        return

    # 2) 답변 생성
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer.convert_tokens_to_ids("<|eot_id|>") else tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]

    results = []
    for sample in tqdm(test_dataset, desc="Inference"):
        sample_id = sample["id"]
        input_ids = sample["prompt"]  # DPO dataset에서는 prompt 사용

        answer_text, reasoning_text = generate_answer(
            model,
            tokenizer,
            input_ids,
            terminators,
            model_args=model_args,
        )

        result = {
            "id": sample_id,
            "output": {
                "answer": answer_text
            },
            "reasoning": reasoning_text if reasoning_text is not None else "",
        }

        results.append(result)

    # 3) 파일 저장
    save_dir = os.path.join(model_dir, "pred_result")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{target_name}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    printi(f"Inference finished. Predictions saved to: {save_path}")


def change_target_dir(target_dir: str, current_stage: str):
    return target_dir + f"_{current_stage}"

def main(
    system_args: SystemArgs,
    model_args: ModelArgs,
    data_args: DataArgs,
    dpo_training_args: DPOTrainingArgs,
    bits_and_bytes_args: BitsAndBytesArgs,
    lora_args: LoraArgs,
    # fsdp_args: FSDPArgs
):
    # 0) 저장 경로 및 백업 설정
    output_dir, target_name = create_out_dir(
        dpo_training_args.output_dir,
        check_sft_type(system_args.use_lora, system_args.use_qlora, lora_args.use_dora, lora_args.use_rslora),
        model_args.model_id.value,
        system_args.additional_info,
        backup_path=system_args.backup_path,
        current_stage=model_args.current_stage
    )
    dpo_training_args.output_dir = output_dir
    dpo_training_args.logging_dir = os.path.join(output_dir, "logs")

    # 1) 설정 파일 업데이트 - DPO용 initialize_config 함수 필요
    bnb_config, lora_config, dpo_training_config = initialize_dpo_config(
        system_args,
        bits_and_bytes_args,
        lora_args,
        dpo_training_args,
        # fsdp_args
    )

    if system_args.train:
        printi("Starting DPO Training")

        # 3) 모델, 토크나이저
        model, tokenizer = prepare_model_tokenmizer(
            model_args=model_args,
            bnb_config=bnb_config,
            is_train=True,  # Training mode
            gradient_checkpointing=dpo_training_config.gradient_checkpointing
        )

        # DPO용 데이터셋 로드 - 기존 data_args의 data_dir 사용
        train_dataset = DPODataset(
            fname=os.path.join(data_args.data_dir, "train.json"),
            tokenizer=tokenizer,
            prompt=model_args.prompt_template if model_args.use_system_prompt else "",
            use_system_prompt=model_args.use_system_prompt,
        )

        eval_dataset = DPODataset(
            fname=os.path.join(data_args.data_dir, "dev.json"),
            tokenizer=tokenizer,
            prompt=model_args.prompt_template if model_args.use_system_prompt else "",
            use_system_prompt=model_args.use_system_prompt,
        )

        # DPO용 데이터셋을 직접 HF Dataset으로 변환
        def prepare_dpo_dataset(dpo_dataset):
            data_dict = {
                "prompt": [],
                "chosen": [],
                "rejected": []
            }

            for item in dpo_dataset:
                # 텍스트 형태로 변환
                prompt_text = tokenizer.decode(item["prompt"], skip_special_tokens=True)
                chosen_text = tokenizer.decode(item["chosen_labels"], skip_special_tokens=True)
                rejected_text = tokenizer.decode(item["rejected_labels"], skip_special_tokens=True)

                data_dict["prompt"].append(prompt_text)
                data_dict["chosen"].append(chosen_text)
                data_dict["rejected"].append(rejected_text)

            return Dataset.from_dict(data_dict)

        train_dataset_hf = prepare_dpo_dataset(train_dataset)
        eval_dataset_hf = prepare_dpo_dataset(eval_dataset)

        # Early Stopping 설정
        early_stopping_callback = [EarlyStoppingCallback(
            early_stopping_patience=model_args.early_stopping,
            early_stopping_threshold=0.0
        )] if model_args.early_stopping != False else None

        # 4) DPO Trainer 설정 - 기존 DPOTrainingArgs를 DPOConfig로 직접 변환
        printi(f"Changed output directory to: {dpo_training_config.output_dir}")
        printi(f"Current stage: {model_args.current_stage}")

        trainer = DPOTrainer(
            model=model,
            args=dpo_training_config,
            train_dataset=train_dataset_hf,
            eval_dataset=eval_dataset_hf,
            processing_class=tokenizer,
            peft_config=lora_config if model_args.current_stage == "" else None,
            callbacks=early_stopping_callback,
            # DPO 전용 파라미터들은 config 객체에서 자동으로 읽어옴
        )

        # 5) DPO 학습 시작
        trainer.train()

        save_training_curves(trainer, dpo_training_config.output_dir)

        # 6) lora 모델 저장
        adapter_dir = os.path.join(dpo_training_config.output_dir, "lora_adapter")
        trainer.model.save_pretrained(adapter_dir)
        printi(f"Model trained and saved to {adapter_dir}")
        printi(f"Training finished. Model saved to: {dpo_training_config.output_dir}")

    # 추론
    if system_args.test:
        run_inference(
            model_dir=dpo_training_config.output_dir,
            model_args=model_args,
            bnb_config=bnb_config,
            target_name=target_name,
            dpo_training_args=dpo_training_args,
        )
        printi(f"Inference completed. Results saved in {dpo_training_config.output_dir}/pred_result")

    print()
    printi("Finished all processes.")


# DPO용 initialize_config 함수 추가
def initialize_dpo_config(
    system_args: SystemArgs,
    bnb_args: BitsAndBytesArgs,
    lora_args: LoraArgs,
    dpo_training_args: DPOTrainingArgs,
    # fsdp_args: FSDPArgs,
):
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig
    from trl import DPOConfig

    bnb_config = None
    lora_config = None

    if system_args.use_qlora:
        bnb_config = BitsAndBytesConfig(**vars(bnb_args))
    if system_args.use_lora:
        lora_config = LoraConfig(**vars(lora_args))

    # DPOConfig에서 지원하지 않는 파라미터들 제거
    dpo_args_dict = vars(dpo_training_args).copy()

    # DPOConfig에서 지원하지 않는 파라미터들 제거
    unsupported_params = [
        'activation_offloading',
        'label_names',
        'beta',
        'loss_type',
        'reference_free',
        'max_prompt_length',
        'max_target_length'
    ]

    # DPO 전용 파라미터들 따로 저장
    dpo_specific_params = {}
    for param in ['beta', 'loss_type', 'reference_free', 'max_length', 'max_prompt_length', 'max_target_length']:
        if param in dpo_args_dict:
            dpo_specific_params[param] = dpo_args_dict[param]

    # 지원하지 않는 파라미터들 제거
    for param in unsupported_params:
        dpo_args_dict.pop(param, None)

    # DPOConfig 생성
    dpo_training_config = DPOConfig(**dpo_args_dict)

    # DPO 전용 파라미터들을 config 객체에 수동으로 추가
    for param, value in dpo_specific_params.items():
        setattr(dpo_training_config, param, value)

    if dpo_training_config.gradient_checkpointing:
        # gradient_checkpointing_kwargs 설정
        dpo_training_config.gradient_checkpointing_kwargs = {
            "use_reentrant": False
        }

    return bnb_config, lora_config, dpo_training_config


if __name__ == "__main__":
    system_args = SystemArgs()
    model_args = ModelArgs()
    data_args = DataArgs()
    dpo_training_args = DPOTrainingArgs()  # SFT -> DPO
    bits_and_bytes_args = BitsAndBytesArgs()
    lora_args = LoraArgs()
    # fsdp_args = FSDPArgs()
    set_seed(system_args.seed)
    os.environ["HF_TOKEN"] = system_args.hf_token
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main(system_args, model_args, data_args, dpo_training_args, bits_and_bytes_args, lora_args) # , fsdp_args)
