import os, json
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import (
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from datasets import Dataset
from trl import SFTTrainer
from peft import PeftModel
from src.configs.config import (
    SystemArgs,
    ModelArgs,
    DataArgs,
    SFTTrainingArgs,
    BitsAndBytesArgs,
    LoraArgs,
)

import random

from sklearn.metrics import accuracy_score

from src.utils.path_utils import create_out_dir
from src.utils.checker import check_sft_type
from src.utils.log_utils import save_training_curves
from src.utils.print_utils import printw, printi, printe
from src.utils.model_utils import (
    initialize_config,
    data_prepare,
    generate_answer,
    prepare_model_tokenmizer,
)


from src.utils.qa_dataset import DataCollatorForSupervisedDataset





def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_inference(
        model_dir: str,
        model_args: ModelArgs,
        bnb_config: BitsAndBytesConfig,
        target_name: str,
        sft_training_args: SFTTrainingArgs,
        data_args: DataArgs,
        use_rag: bool = False
):
    printi("Starting Inference")

    # 1) 모델 및 토크나이저 로드
    model, tokenizer = prepare_model_tokenmizer(
        model_args=model_args,
        bnb_config=bnb_config,
        is_train=False,
        gradient_checkpointing=sft_training_args.gradient_checkpointing
    )

    data_dict = data_prepare(
            ["test"],
            data_args,
            model_args,
            tokenizer,
            use_rag=use_rag,
            is_test_and_drop_other_info=model_args.is_test_and_drop_other_info
        )

    adapter_dir = os.path.join(model_dir, model_args.load_model)
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
    )

    # 2) 답변 생성
    # terminators = [
    #     tokenizer.eos_token_id,
    #     tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer.convert_tokens_to_ids("<|eot_id|>") else tokenizer.convert_tokens_to_ids("<|endoftext|>")
    # ]


    results = []
    for sample in tqdm(data_dict["test"], desc="Inference"):
        input_ids = sample["input_ids"]
        original_data = sample["original_data"]

        answer_text, reasoning_text = generate_answer(
            model,
            tokenizer,
            input_ids,
            # terminators,
            model_args=model_args,
        )

        original_data["output"] = {
            "answer": answer_text,
        }
        original_data["reasoning"] = reasoning_text if reasoning_text is not None else ""

        printw(f"{answer_text}")

        results.append(original_data)

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
    sft_training_args: SFTTrainingArgs,
    bits_and_bytes_args: BitsAndBytesArgs,
    lora_args: LoraArgs,
):
    # DataArgs에 use_rag 설정 전달
    data_args.set_use_rag(system_args.use_rag)

    # 0) 저장 경로 및 백업 설정

    output_dir, target_name = create_out_dir(
        sft_training_args.output_dir,
        check_sft_type(system_args.use_lora, system_args.use_qlora, lora_args.use_dora, lora_args.use_rslora),
        model_args.model_id.value,
        system_args.additional_info,
        backup_path=system_args.backup_path,
        current_stage=model_args.current_stage
    )
    sft_training_args.output_dir = output_dir
    sft_training_args.logging_dir = os.path.join(output_dir, "logs")

    # 1) 설정 파일 업데이트
    bnb_config, lora_config, sft_training_config = initialize_config(
        system_args,
        bits_and_bytes_args,
        lora_args,
        sft_training_args,
    )

    if system_args.train:
        printi("Starting Training")
        printi(f"Using RAG: {system_args.use_rag}")
        printi(f"Data directory: {data_args.data_dir}")

        # 3) 모델, 토크나이저
        model, tokenizer = prepare_model_tokenmizer(
            model_args=model_args,
            bnb_config=bnb_config,
            is_train=True,  # Training mode
            gradient_checkpointing=sft_training_config.gradient_checkpointing
        )
        print("model.config.max_position_embeddings", model.config.max_position_embeddings)

        data_dict = data_prepare(
            ["train", "dev"],
            data_args,
            model_args,
            tokenizer,
            use_rag=system_args.use_rag,
            is_test_and_drop_other_info=model_args.is_test_and_drop_other_info  # 학습 모드로 설정
        )

        train_dataset_hf = Dataset.from_dict({
            "input_ids": [item['input_ids'] for item in data_dict["train"]],
            "labels": [item['labels'] for item in data_dict["train"]],
        })

        eval_dataset_hf = Dataset.from_dict({
            "input_ids": [item['input_ids'] for item in data_dict["dev"]],
            "labels": [item['labels'] for item in data_dict["dev"]],
        })

        # Early Stopping 설정
        early_stopping_callback = [EarlyStoppingCallback(
            early_stopping_patience=model_args.early_stopping,
            early_stopping_threshold=0.0
        )] if model_args.early_stopping != False else None

        # 4) Trainer 설정
        printi(f"Changed output directory to: {sft_training_config.output_dir}")
        printi(f"Current stage: {model_args.current_stage}")

        trainer = SFTTrainer(
            model=model,
            args=sft_training_config,
            train_dataset=train_dataset_hf,
            eval_dataset=eval_dataset_hf,
            peft_config=lora_config if model_args.current_stage == "" else None,
            preprocess_logits_for_metrics=logits_to_cpu,
            data_collator=DataCollatorForSupervisedDataset(tokenizer),
            callbacks=early_stopping_callback
        )

        # 이제 실제 학습 시작
        trainer.train()

        save_training_curves(trainer, sft_training_config.output_dir)

        # 6) lora 모델 저장
        adapter_dir = os.path.join(sft_training_config.output_dir, "lora_adapter")
        trainer.model.save_pretrained(adapter_dir)
        printi(f"Model trained and saved to {adapter_dir}")
        printi(f"Training finished. Model saved to: {sft_training_config.output_dir}")

    # 추론
    if system_args.test:
        run_inference(
            model_dir=sft_training_config.output_dir,
            model_args=model_args,
            bnb_config=bnb_config,
            target_name=target_name,
            sft_training_args=sft_training_args,
            data_args=data_args,
            use_rag=system_args.use_rag
        )
        printi(f"Inference completed. Results saved in {sft_training_config.output_dir}/pred_result")

    print()
    printi("Finished all processes.")


def logits_to_cpu(logits, labels):
    return logits.cpu(), labels.cpu()


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
