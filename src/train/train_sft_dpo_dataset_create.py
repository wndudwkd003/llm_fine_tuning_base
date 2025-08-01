import os, json
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import (
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig
from peft import PeftModel
from src.configs.config import (
    SystemArgs,
    ModelArgs,
    DataArgs,
    SFTTrainingArgs,
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
from src.utils.dpo_dataset_utils import create_dpo_dataset

LABEL_PAD_TOKEN_ID = -100


def run_inference(
        test_dataset,
        model_dir: str,
        model_args: ModelArgs,
        bnb_config: BitsAndBytesConfig,
        target_name: str,
        sft_training_args: SFTTrainingArgs
):
    printi("Starting Inference")

    # 1) 모델 및 토크나이저 로드
    model, tokenizer = prepare_model_tokenmizer(
        model_args=model_args,
        bnb_config=bnb_config,
        is_train=False,
        gradient_checkpointing=sft_training_args.gradient_checkpointing
    )

    adapter_dir = os.path.join(model_dir, model_args.load_model)
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
    )

    # 2) 답변 생성
    eot_token_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
    terminators = [tokenizer.eos_token_id] if eot_token_id is None else [tokenizer.eos_token_id, eot_token_id]

    results = []
    for example in tqdm(test_dataset, desc="Inference"):
        prompt_ids = tokenizer.apply_chat_template(
            conversation=example['messages'],
            tokenize=True,
            truncation=True,
            padding="max_length",
            max_length=sft_training_args.max_length,
            add_generation_prompt=True,
            return_tensors="pt"
        )

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
    save_dir = os.path.join(model_dir, "pred_result")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{target_name}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    printi(f"Inference finished. Predictions saved to: {save_path}")


def main(
    system_args: SystemArgs,
    model_args: ModelArgs,
    data_args: DataArgs,
    sft_training_args: SFTTrainingArgs,
    bits_and_bytes_args: BitsAndBytesArgs,
    lora_args: LoraArgs,
    # fsdp_args: FSDPArgs
):
    global LABEL_PAD_TOKEN_ID
    LABEL_PAD_TOKEN_ID = data_args.label_pad_token_id

    # 0) 저장 경로 및 백업 설정
    output_dir, target_name = create_out_dir(
        sft_training_args.output_dir,
        check_sft_type(system_args.use_lora, system_args.use_qlora),
        model_args.model_id.value,
        system_args.additional_info,
        backup_path=system_args.backup_path
    )
    sft_training_args.output_dir = output_dir
    sft_training_args.logging_dir = os.path.join(output_dir, "logs")

    # 1) 설정 파일 업데이트
    bnb_config, lora_config, sft_training_config = initialize_config(
        system_args,
        bits_and_bytes_args,
        lora_args,
        sft_training_args,
        # fsdp_args
    )


    # DPO 데이터셋 생성 모드
    if system_args.dpo_dataset_create_mode:
        printi("Running in DPO dataset creation mode.")

        # 1. 새로운 저장 경로 생성
        dpo_data_dir = data_args.data_dir + "_for_dpo"
        os.makedirs(dpo_data_dir, exist_ok=True)
        printi(f"DPO datasets will be saved in: {dpo_data_dir}")

        # 2. 추론에 사용할 모델과 토크나이저 로드
        model, tokenizer = prepare_model_tokenmizer(
            model_args=model_args,
            bnb_config=bnb_config,
            is_train=False,
            gradient_checkpointing=sft_training_args.gradient_checkpointing
        )
        adapter_dir = os.path.join(sft_training_args.output_dir, model_args.load_model)
        if os.path.exists(adapter_dir):
            printi(f"Loading adapter from: {adapter_dir}")
            model = PeftModel.from_pretrained(model, adapter_dir)

        # 3. 원본 데이터셋(train, dev) 로드
        data_dict = data_prepare(
            ["train", "dev"],
            data_args,
            system_args,
            sft_training_args,
            model_args
        )

        # 4. train, dev 데이터셋에 대해 DPO 데이터 생성
        for split in ["train", "dev"]:
            output_file = os.path.join(dpo_data_dir, f"{split}.json")
            create_dpo_dataset(
                source_dataset=data_dict[split],
                model=model,
                tokenizer=tokenizer,
                model_args=model_args,
                sft_training_args=sft_training_args,
                output_path=output_file
            )

        # 5. test.json 복사
        original_test_path = os.path.join(data_args.data_dir, "test.json")
        dpo_test_path = os.path.join(dpo_data_dir, "test.json")
        if os.path.exists(original_test_path):
            import shutil
            shutil.copy(original_test_path, dpo_test_path)
            printi(f"Copied original test set to: {dpo_test_path}")
        else:
            printw(f"Original test set not found at: {original_test_path}")


        printi("Finished DPO dataset creation.")
        return # 데이터 생성 후 종료


    if system_args.train:
        data_dict = data_prepare(
            ["train", "dev"],
            data_args,
            system_args,
            sft_training_args,
            model_args
        )

        printi("Starting Training")

        # 3) 모델, 토크나이저
        model, tokenizer = prepare_model_tokenmizer(
            model_args=model_args,
            bnb_config=bnb_config,
            is_train=True,  # Training mode
            gradient_checkpointing=sft_training_config.gradient_checkpointing
        )


        # 4) Trainer 설정
        trainer = SFTTrainer(
            model=model,
            args=sft_training_config,
            train_dataset=data_dict["train"],
            eval_dataset=data_dict["dev"],
            processing_class=tokenizer,
            peft_config=lora_config,
            preprocess_logits_for_metrics=logits_to_cpu,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=model_args.early_stopping,
                early_stopping_threshold=0.0
            )]
        )

        # 5) 모델 학습
        trainer.train()
        save_training_curves(trainer, sft_training_config.output_dir)

        # 6) lora 모델 저장
        adapter_dir = os.path.join(sft_training_config.output_dir, "lora_adapter")
        trainer.model.save_pretrained(adapter_dir)
        printi(f"Model trained and saved to {adapter_dir}")
        printi(f"Training finished. Model saved to: {sft_training_config.output_dir}")

    # 5) 추론 (system_args.test가 True일 때 실행)
    if system_args.test:
        data_dict = data_prepare(
            ["test"],
            data_args,
            system_args,
            sft_training_args,
            model_args
        )

        run_inference(
            test_dataset=data_dict["test"],
            model_dir=sft_training_config.output_dir,
            model_args=model_args,
            bnb_config=bnb_config,
            target_name=target_name,
            sft_training_args=sft_training_args
        )
        printi(f"Inference completed. Results saved in {sft_training_config.output_dir}/pred_result/{target_name}.json")

    print()
    printi("Finished all processes.")


def logits_to_cpu(logits, labels):
    preds = torch.argmax(logits, dim=-1).cpu().to(torch.int32)
    return preds, labels.cpu()

def compute_metrics(eval_preds):
    preds_nested, labels_nested = eval_preds  # object 배열 또는 (steps, batch, seq)

    preds  = np.concatenate([np.asarray(p).ravel() for p in preds_nested])
    labels = np.concatenate([np.asarray(l).ravel() for l in labels_nested])

    if preds.size != labels.size:
        min_len = min(preds.size, labels.size)
        preds  = preds[:min_len]
        labels = labels[:min_len]

    mask = labels != LABEL_PAD_TOKEN_ID
    if not mask.any():
        return {"accuracy": 0.0}

    acc = accuracy_score(labels[mask], preds[mask])
    return {"accuracy": acc}


if __name__ == "__main__":
    system_args = SystemArgs()
    model_args = ModelArgs()
    data_args = DataArgs()
    sft_training_args = SFTTrainingArgs()
    bits_and_bytes_args = BitsAndBytesArgs()
    lora_args = LoraArgs()
    # fsdp_args = FSDPArgs()
    set_seed(system_args.seed)
    os.environ["HF_TOKEN"] = system_args.hf_token
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main(system_args, model_args, data_args, sft_training_args, bits_and_bytes_args, lora_args) # , fsdp_args)
