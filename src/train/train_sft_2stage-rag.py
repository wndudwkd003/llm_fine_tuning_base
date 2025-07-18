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
    prepare_model_tokenmizer,
)

# 새로운 2단계 RAG 데이터셋 import
from src.utils.two_stage_rag_dataset import TwoStageRAGDataset, DataCollatorForSupervisedDataset


def train_query_generation_stage(
    system_args: SystemArgs,
    model_args: ModelArgs,
    data_args: DataArgs,
    sft_training_args: SFTTrainingArgs,
    bits_and_bytes_args: BitsAndBytesArgs,
    lora_args: LoraArgs,
    output_dir: str
):
    """1단계: 검색 질의 생성 학습"""
    printi("="*50)
    printi("STAGE 1: Query Generation Training")
    printi("="*50)

    # 1단계 전용 출력 디렉토리
    stage1_output_dir = os.path.join(output_dir, "stage1_query_generation")
    os.makedirs(stage1_output_dir, exist_ok=True)

    # 1단계 전용 설정 복사
    stage1_sft_args = sft_training_args
    stage1_sft_args.output_dir = stage1_output_dir
    stage1_sft_args.logging_dir = os.path.join(stage1_output_dir, "logs")

    # 설정 초기화
    bnb_config, lora_config, sft_training_config = initialize_config(
        system_args,
        bits_and_bytes_args,
        lora_args,
        stage1_sft_args,
    )

    # 모델, 토크나이저 준비
    model, tokenizer = prepare_model_tokenmizer(
        model_args=model_args,
        bnb_config=bnb_config,
        is_train=True,
        gradient_checkpointing=sft_training_config.gradient_checkpointing
    )

    # 1단계 데이터셋 생성
    train_dataset = TwoStageRAGDataset(
        fname=os.path.join(data_args.data_dir, "train.json"),
        tokenizer=tokenizer,
        stage="query_generation"
    )

    eval_dataset = TwoStageRAGDataset(
        fname=os.path.join(data_args.data_dir, "dev.json"),
        tokenizer=tokenizer,
        stage="query_generation"
    )

    # Hugging Face Dataset으로 변환
    train_dataset_hf = Dataset.from_dict({
        "input_ids": [item['input_ids'] for item in train_dataset],
        "labels": [item['labels'] for item in train_dataset],
    })

    eval_dataset_hf = Dataset.from_dict({
        "input_ids": [item['input_ids'] for item in eval_dataset],
        "labels": [item['labels'] for item in eval_dataset],
    })

    # Early Stopping 설정
    early_stopping_callback = [EarlyStoppingCallback(
        early_stopping_patience=model_args.early_stopping,
        early_stopping_threshold=0.0
    )] if model_args.early_stopping != False else None

    # 1단계 Trainer 설정
    trainer = SFTTrainer(
        model=model,
        args=sft_training_config,
        train_dataset=train_dataset_hf,
        eval_dataset=eval_dataset_hf,
        peft_config=lora_config,  # 새로운 LoRA adapter 생성
        preprocess_logits_for_metrics=logits_to_cpu,
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
        callbacks=early_stopping_callback
    )

    # 1단계 학습 시작
    printi("Starting Query Generation Training...")
    trainer.train()

    # 학습 곡선 저장
    save_training_curves(trainer, sft_training_config.output_dir)

    # 1단계 adapter 저장
    adapter_dir = os.path.join(sft_training_config.output_dir, "lora_adapter")
    trainer.model.save_pretrained(adapter_dir)
    printi(f"Query generation model saved to {adapter_dir}")

    return stage1_output_dir


def train_answer_generation_stage(
    system_args: SystemArgs,
    model_args: ModelArgs,
    data_args: DataArgs,
    sft_training_args: SFTTrainingArgs,
    bits_and_bytes_args: BitsAndBytesArgs,
    lora_args: LoraArgs,
    output_dir: str,
    stage1_output_dir: str
):
    """2단계: 답변 생성 학습 (1단계 adapter 병합 후 새 adapter 생성)"""
    printi("="*50)
    printi("STAGE 2: Answer Generation Training")
    printi("="*50)

    # 2단계 전용 출력 디렉토리
    stage2_output_dir = os.path.join(output_dir, "stage2_answer_generation")
    os.makedirs(stage2_output_dir, exist_ok=True)

    # 2단계 전용 설정 복사
    stage2_sft_args = sft_training_args
    stage2_sft_args.output_dir = stage2_output_dir
    stage2_sft_args.logging_dir = os.path.join(stage2_output_dir, "logs")

    # 설정 초기화
    bnb_config, lora_config, sft_training_config = initialize_config(
        system_args,
        bits_and_bytes_args,
        lora_args,
        stage2_sft_args,
    )

    # 기본 모델, 토크나이저 준비
    model, tokenizer = prepare_model_tokenmizer(
        model_args=model_args,
        bnb_config=bnb_config,
        is_train=True,
        gradient_checkpointing=sft_training_config.gradient_checkpointing
    )

    # 1단계 adapter 로드 및 병합
    stage1_adapter_dir = os.path.join(stage1_output_dir, "lora_adapter")
    if os.path.exists(stage1_adapter_dir):
        printi(f"Loading query generation adapter from {stage1_adapter_dir}")

        # 1단계 adapter 로드
        model = PeftModel.from_pretrained(model, stage1_adapter_dir)

        # 병합 (merge and unload)
        model = model.merge_and_unload()
        printi("Query generation adapter merged with base model")
    else:
        printe(f"Stage 1 adapter not found at {stage1_adapter_dir}")
        raise FileNotFoundError(f"Stage 1 adapter not found at {stage1_adapter_dir}")

    # 2단계 데이터셋 생성
    train_dataset = TwoStageRAGDataset(
        fname=os.path.join(data_args.data_dir, "train.json"),
        tokenizer=tokenizer,
        stage="answer_generation"
    )

    eval_dataset = TwoStageRAGDataset(
        fname=os.path.join(data_args.data_dir, "dev.json"),
        tokenizer=tokenizer,
        stage="answer_generation"
    )

    # Hugging Face Dataset으로 변환
    train_dataset_hf = Dataset.from_dict({
        "input_ids": [item['input_ids'] for item in train_dataset],
        "labels": [item['labels'] for item in train_dataset],
    })

    eval_dataset_hf = Dataset.from_dict({
        "input_ids": [item['input_ids'] for item in eval_dataset],
        "labels": [item['labels'] for item in eval_dataset],
    })

    # Early Stopping 설정
    early_stopping_callback = [EarlyStoppingCallback(
        early_stopping_patience=model_args.early_stopping,
        early_stopping_threshold=0.0
    )] if model_args.early_stopping != False else None

    # 2단계 Trainer 설정 (병합된 모델 위에 새로운 LoRA adapter 생성)
    trainer = SFTTrainer(
        model=model,
        args=sft_training_config,
        train_dataset=train_dataset_hf,
        eval_dataset=eval_dataset_hf,
        peft_config=lora_config,  # 새로운 LoRA adapter 생성
        preprocess_logits_for_metrics=logits_to_cpu,
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
        callbacks=early_stopping_callback
    )

    # 2단계 학습 시작
    printi("Starting Answer Generation Training...")
    trainer.train()

    # 학습 곡선 저장
    save_training_curves(trainer, sft_training_config.output_dir)

    # 2단계 adapter 저장
    adapter_dir = os.path.join(sft_training_config.output_dir, "lora_adapter")
    trainer.model.save_pretrained(adapter_dir)
    printi(f"Answer generation model saved to {adapter_dir}")

    return stage2_output_dir


def run_two_stage_inference(
    model_dir: str,
    model_args: ModelArgs,
    bnb_config: BitsAndBytesConfig,
    target_name: str,
    sft_training_args: SFTTrainingArgs,
    data_args: DataArgs,
    rag_retriever=None
):
    """2단계 RAG 추론"""
    printi("Starting Two-Stage RAG Inference")

    # 1단계 모델 로드 (검색 질의 생성)
    query_gen_model, tokenizer = prepare_model_tokenmizer(
        model_args=model_args,
        bnb_config=bnb_config,
        is_train=False,
        gradient_checkpointing=sft_training_args.gradient_checkpointing
    )

    stage1_adapter_dir = os.path.join(model_dir, "stage1_query_generation", "lora_adapter")
    query_gen_model = PeftModel.from_pretrained(query_gen_model, stage1_adapter_dir)

    # 2단계 모델 로드 (답변 생성)
    answer_gen_model, _ = prepare_model_tokenmizer(
        model_args=model_args,
        bnb_config=bnb_config,
        is_train=False,
        gradient_checkpointing=sft_training_args.gradient_checkpointing
    )

    # 1단계 adapter 병합
    answer_gen_model = PeftModel.from_pretrained(answer_gen_model, stage1_adapter_dir)
    answer_gen_model = answer_gen_model.merge_and_unload()

    # 2단계 adapter 로드
    stage2_adapter_dir = os.path.join(model_dir, "stage2_answer_generation", "lora_adapter")
    answer_gen_model = PeftModel.from_pretrained(answer_gen_model, stage2_adapter_dir)

    # 테스트 데이터 로드
    test_dataset = TwoStageRAGDataset(
        fname=os.path.join(data_args.data_dir, "test.json"),
        tokenizer=tokenizer,
        stage="query_generation"
    )

    results = []
    for idx in tqdm(range(len(test_dataset)), desc="Two-Stage Inference"):
        sample = test_dataset[idx]
        input_ids = sample["input_ids"]
        original_data = sample["original_data"]

        # 1단계: 검색 질의 3개 생성
        query_response, _ = generate_answer(
            query_gen_model,
            tokenizer,
            input_ids,
            model_args=model_args,
        )

        # 질의 추출
        from src.utils.two_stage_rag_dataset import extract_queries_from_response
        queries = extract_queries_from_response(query_response)

        # 원래 질문 추가
        original_question = original_data["input"]["question"]
        all_queries = [original_question] + queries

        # RAG 검색 (각 질의당 top_k=2)
        all_contexts = []
        if rag_retriever:
            for query in all_queries:
                contexts = rag_retriever.search(query, top_k=2)
                all_contexts.extend(contexts)

        # 중복 제거 및 상위 4개 선택
        from src.utils.two_stage_rag_dataset import deduplicate_and_rank_contexts
        selected_contexts = deduplicate_and_rank_contexts(all_contexts, max_contexts=4)

        # 2단계: 최종 답변 생성을 위한 새로운 프롬프트 구성
        from src.utils.two_stage_rag_dataset import TYPE_INSTRUCTIONS
        instruction = TYPE_INSTRUCTIONS.get(original_data["input"]["question_type"], "")

        chat_parts = [instruction]

        if selected_contexts:
            context_text = "[참고 문서]\n"
            for i, ctx in enumerate(selected_contexts, 1):
                context_text += f"{i}. {ctx['text']}\n"
            chat_parts.append(context_text)

        chat_parts.append(f"[질문] {original_question}")
        final_prompt = " ".join(chat_parts)

        # 토크나이징
        message = [{"role": "user", "content": final_prompt}]
        final_input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        # 최종 답변 생성
        final_answer, reasoning_text = generate_answer(
            answer_gen_model,
            tokenizer,
            final_input_ids,
            model_args=model_args,
        )

        # 결과 저장
        original_data["output"] = {
            "answer": final_answer,
            "rag_queries": queries,
            "raw_response": query_response
        }
        original_data["retrieved_contexts"] = selected_contexts
        original_data["reasoning"] = reasoning_text if reasoning_text is not None else ""

        printw(f"Generated queries: {queries}")
        printw(f"Final answer: {final_answer}")

        results.append(original_data)

    # 결과 저장
    save_dir = os.path.join(model_dir, "pred_result")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{target_name}_two_stage.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    printi(f"Two-stage inference finished. Predictions saved to: {save_path}")


def main(
    system_args: SystemArgs,
    model_args: ModelArgs,
    data_args: DataArgs,
    sft_training_args: SFTTrainingArgs,
    bits_and_bytes_args: BitsAndBytesArgs,
    lora_args: LoraArgs,
):
    # 기본 설정
    system_args.use_rag = True  # 2단계 RAG 사용

    # 출력 디렉토리 설정
    rag_suffix = "_two_stage_rag"
    output_dir, target_name = create_out_dir(
        sft_training_args.output_dir,
        check_sft_type(system_args.use_lora, system_args.use_qlora, lora_args.use_dora, lora_args.use_rslora),
        model_args.model_id.value,
        system_args.additional_info + rag_suffix,
        backup_path=system_args.backup_path,
        current_stage="two_stage"
    )

    if system_args.train:
        printi("Starting Two-Stage RAG Training")
        printi(f"Data directory: {data_args.data_dir}")
        printi(f"Output directory: {output_dir}")

        # 1단계: 검색 질의 생성 학습
        stage1_output_dir = train_query_generation_stage(
            system_args, model_args, data_args, sft_training_args,
            bits_and_bytes_args, lora_args, output_dir
        )

        # 2단계: 답변 생성 학습 (1단계 adapter 병합 후 새 adapter 생성)
        stage2_output_dir = train_answer_generation_stage(
            system_args, model_args, data_args, sft_training_args,
            bits_and_bytes_args, lora_args, output_dir, stage1_output_dir
        )

        printi("="*50)
        printi("Two-Stage RAG Training Completed!")
        printi(f"Stage 1 model: {stage1_output_dir}")
        printi(f"Stage 2 model: {stage2_output_dir}")
        printi("="*50)

    # 추론
    if system_args.test:
        printi("Starting Two-Stage RAG Inference")

        # 실제 RAG 검색 시스템 연결 필요
        # rag_retriever = YourRAGRetriever()

        run_two_stage_inference(
            model_dir=output_dir,
            model_args=model_args,
            bnb_config=None,
            target_name=target_name,
            sft_training_args=sft_training_args,
            data_args=data_args,
            rag_retriever=None  # 실제 RAG 시스템으로 교체
        )

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
