import os, json
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import BitsAndBytesConfig
from peft import PeftModel
from src.configs.config import (
    SystemArgs,
    ModelArgs,
    DataArgs,
    SFTTrainingArgs,
    BitsAndBytesArgs,
    LoraArgs,
)

from src.utils.seeds import set_seed
from src.utils.path_utils import create_out_dir
from src.utils.checker import check_sft_type
from src.utils.print_utils import printw, printi, printe
from src.utils.model_utils import (
    initialize_config,
    data_prepare,
    generate_answer,
    prepare_model_tokenmizer
)


def load_original_data(dataset_path, dataset_name):
    """원본 데이터셋 파일 로드"""
    file_path = os.path.join(dataset_path, f"{dataset_name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        printi(f"Loaded {len(data)} samples from {file_path}")
        return data
    else:
        printi(f"File not found: {file_path}")
        return []


def update_original_data_with_predictions(original_data, predictions):
    """원본 데이터의 answer를 모델 예측으로 업데이트"""
    # 예측 결과를 id로 매핑
    pred_dict = {pred["id"]: pred for pred in predictions}

    updated_data = []
    matched_count = 0

    for item in original_data:
        item_id = item["id"]
        if item_id in pred_dict:
            # 원본 구조 유지하면서 answer만 업데이트
            updated_item = item.copy()
            updated_item["output"]["answer"] = pred_dict[item_id]["output"]["answer"]

            # reasoning이 있다면 추가
            if pred_dict[item_id]["reasoning"]:
                updated_item["reasoning"] = pred_dict[item_id]["reasoning"]

            updated_data.append(updated_item)
            matched_count += 1
        else:
            # 매칭되지 않은 경우 원본 유지
            updated_data.append(item)

    printi(f"Matched {matched_count} out of {len(original_data)} samples")
    return updated_data


def run_inference(
        model_dir: str,
        model_args: ModelArgs,
        bnb_config: BitsAndBytesConfig,
        target_name: str,
        sft_training_args: SFTTrainingArgs,
        original_dataset_path: str,
        target_dataset_path: str
):
    printi("Starting Inference")

    os.makedirs(target_dataset_path, exist_ok=True)

    # 1) 모델 및 토크나이저 로드
    model, tokenizer = prepare_model_tokenmizer(
        model_args=model_args,
        bnb_config=bnb_config,
        is_train=False,
        gradient_checkpointing=sft_training_args.gradient_checkpointing
    )

    # 모든 데이터셋 로드 (train, dev, test)
    data_dict = data_prepare(
            ["train", "dev"],
            data_args,
            model_args,
            tokenizer
        )

    adapter_dir = os.path.join(model_dir, model_args.load_model)
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
    )

    # 2) 답변 생성
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer.convert_tokens_to_ids("<|eot_id|>") else tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]

    # 각 데이터셋에 대해 추론 수행
    for dataset_name in ["train", "dev"]:
        if dataset_name not in data_dict:
            printi(f"Dataset {dataset_name} not found. Skipping...")
            continue

        printi(f"Processing {dataset_name} dataset...")

        # 모델 예측 수행
        updated_data = []
        for sample in tqdm(data_dict[dataset_name], desc=f"Inference on {dataset_name}"):
            sample_id = sample["id"]
            input_ids = sample["input_ids"]
            original_data = sample["original_data"]

            answer_text, reasoning_text = generate_answer(
                model,
                tokenizer,
                input_ids,
                terminators,
                model_args=model_args,
            )

            updated_item = original_data.copy()
            updated_item["output"]["answer"] = answer_text

            if reasoning_text:
                updated_item["reasoning"] = reasoning_text


            updated_data.append(updated_item)
            print("updated_item:", updated_item)

        # 3) 원본 데이터 로드 및 업데이트
        save_path = os.path.join(target_dataset_path, f"{dataset_name}.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)

        printi(f"Updated {dataset_name} data saved to: {save_path}")


def main(
    system_args: SystemArgs,
    model_args: ModelArgs,
    data_args: DataArgs,
    sft_training_args: SFTTrainingArgs,
    bits_and_bytes_args: BitsAndBytesArgs,
    lora_args: LoraArgs,
):
    # 0) 저장 경로 및 백업 설정
    output_dir, target_name = create_out_dir(
        sft_training_args.output_dir,
        check_sft_type(system_args.use_lora, system_args.use_qlora, lora_args.use_dora),
        model_args.model_id.value,
        system_args.additional_info,
        backup_path=system_args.backup_path,
        current_stage=model_args.current_stage
    )

    # 폴더 이름에 "*cot*_reject_1-stage" 추가
    # 원본 데이터셋 경로 설정
    original_dataset_path = "datasets/sub_3_data_korean_culture_qa_V1.0_preprocessed"
    target_dataset_path =  original_dataset_path + "_cot_reject_1-stage"

    printi(f"Inference mode: Updated output directory to: {output_dir}")

    sft_training_args.output_dir = output_dir
    sft_training_args.logging_dir = os.path.join(output_dir, "logs")

    # 1) 설정 파일 업데이트
    bnb_config, lora_config, sft_training_config = initialize_config(
        system_args,
        bits_and_bytes_args,
        lora_args,
        sft_training_args,
    )


    # 추론 실행
    run_inference(
        model_dir=sft_training_config.output_dir,
        model_args=model_args,
        bnb_config=bnb_config,
        target_name=target_name,
        sft_training_args=sft_training_args,
        original_dataset_path=original_dataset_path,
        target_dataset_path=target_dataset_path,
    )

    printi(f"Inference completed. Results saved in {sft_training_config.output_dir}/pred_result")
    printi("Finished all processes.")


if __name__ == "__main__":
    system_args = SystemArgs()
    model_args = ModelArgs()
    data_args = DataArgs()
    sft_training_args = SFTTrainingArgs()
    bits_and_bytes_args = BitsAndBytesArgs()
    lora_args = LoraArgs()

    set_seed(system_args.seed)
    os.environ["HF_TOKEN"] = system_args.hf_token
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    main(system_args, model_args, data_args, sft_training_args, bits_and_bytes_args, lora_args)
