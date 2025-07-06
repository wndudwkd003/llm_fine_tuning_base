import os, json, random
import torch
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel, PeftModelForCausalLM
from src.configs.config import (
    SystemArgs,
    ModelArgs,
    DataArgs,
    SFTTrainingArgs,
    BitsAndBytesArgs,
    LoraArgs
)

from sklearn.metrics import accuracy_score
from src.utils.path_utils import create_out_dir
from src.utils.checker import check_sft_type
from src.utils.log_utils import save_training_curves
from src.utils.print_utils import printw, printi, printe

LABEL_PAD_TOKEN_ID = -100

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_config(
    system_args: SystemArgs,
    bnb_args: BitsAndBytesArgs,
    lora_args: LoraArgs,
    sft_training_args: SFTTrainingArgs
):
    bnb_config = None
    lora_config = None

    if system_args.use_qlora:
        bnb_config = BitsAndBytesConfig(**vars(bnb_args))
    if system_args.use_lora:
        lora_config = LoraConfig(**vars(lora_args))

    sft_training_config = SFTConfig(**vars(sft_training_args))
    return bnb_config, lora_config, sft_training_config


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

    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(prompt_ids)

    outputs = model.generate(
        input_ids=prompt_ids.unsqueeze(0),
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
        bnb_config: BitsAndBytesConfig,
        target_name: str
):
    printi("Starting Inference")

    # 1) 모델 및 토크나이저 로드 (기존과 동일)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id.value,
        attn_implementation="flash_attention_2" if model_args.use_flash_attn2 else "eager",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=model_args.dtype.value,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    adapter_dir = os.path.join(model_dir, "lora_adapter")
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
    )

    model.eval()
    model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_id.value,
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

def data_prepare(data_splits, data_args: DataArgs):# -> Dataset | DatasetDict | Any:
    data_files = {sp: os.path.join(data_args.data_dir, f"{sp}.json") for sp in data_splits}
    data_dict = load_dataset("json",data_files=data_files, num_proc=system_args.num_proc)
    data_dict = data_dict.map(
        lambda example: add_system_prompt(example, system_prompt=model_args.prompt_template),
        num_proc=system_args.num_proc
    )
    return data_dict

def main(
    system_args: SystemArgs,
    model_args: ModelArgs,
    data_args: DataArgs,
    sft_training_args: SFTTrainingArgs,
    bits_and_bytes_args: BitsAndBytesArgs,
    lora_args: LoraArgs
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
        sft_training_args
    )

    if system_args.train:
        data_dict = data_prepare(["train", "dev"], data_args)
        printi("Starting Training")

        # 3) 모델, 토크나이저
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_id.value,
            trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_id.value,
            torch_dtype=model_args.dtype.value,
            attn_implementation="flash_attention_2" if model_args.use_flash_attn2 else "eager",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        model.config.use_cache = False
        # model.gradient_checkpointing_enable()

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
        )

        # 5) 모델 학습
        trainer.train()
        save_training_curves(trainer, sft_training_config.output_dir)

        # 6) lora 모델 저장
        adapter_dir = os.path.join(sft_training_config.output_dir, "lora_adapter")
        trainer.model.save_pretrained(adapter_dir)
        printi(f"Model trained and saved to {adapter_dir}")

        del trainer.model
        del trainer
        torch.cuda.empty_cache()

    # 5) 추론 (system_args.test가 True일 때 실행)
    if system_args.test:
        data_dict = data_prepare(["test"], data_args)
        run_inference(
            test_dataset=data_dict["test"],
            model_dir=sft_training_config.output_dir,
            model_args=model_args,
            bnb_config=bnb_config,
            target_name=target_name
        )

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
    set_seed(system_args.seed)
    os.environ["HF_TOKEN"] = system_args.hf_token
    main(system_args, model_args, data_args, sft_training_args, bits_and_bytes_args, lora_args)
