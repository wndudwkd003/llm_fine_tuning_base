import os
import torch
import re
from dataclasses import asdict
from transformers import BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTConfig
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from src.configs.config import (
    SystemArgs,
    SFTTrainingArgs,
    BitsAndBytesArgs,
    LoraArgs,
    DataArgs,
    ModelArgs,
    # FSDPArgs
)
from src.utils.print_utils import printi
from src.utils.qa_dataset import CustomDataset

def initialize_config(
    system_args: SystemArgs,
    bnb_args: BitsAndBytesArgs,
    lora_args: LoraArgs,
    sft_training_args: SFTTrainingArgs,
    # fsdp_args: FSDPArgs,
):
    bnb_config = None
    lora_config = None

    if system_args.use_qlora:
        bnb_config = BitsAndBytesConfig(**vars(bnb_args))
    if system_args.use_lora:
        lora_config = LoraConfig(**vars(lora_args))

    # sft_training_args.fsdp_config = asdict(fsdp_args)
    sft_training_config = SFTConfig(**vars(sft_training_args))

    if sft_training_config.fsdp and sft_training_config.gradient_checkpointing:
        # gradient_checkpointing_kwargs 설정
        sft_training_config.gradient_checkpointing_kwargs = {
            "use_reentrant": False
        }


    return bnb_config, lora_config, sft_training_config


def add_system_prompt(example, system_prompt: str, use_system_prompt: bool = True):
    if use_system_prompt:
        example['messages'] = [{"role": "system", "content": system_prompt}] + example['messages']
    else:
        example['messages'][0]['content'] = system_prompt + "\n" + example['messages'][0]['content']
    return example


# def data_prepare(
#     data_splits,
#     data_args: DataArgs,
#     system_args: SystemArgs,
#     sft_training_args: SFTTrainingArgs,
#     model_args: ModelArgs,
#     tokenizer: AutoTokenizer,
# ):
#     data_files = {sp: os.path.join(data_args.data_dir, f"{sp}.json") for sp in data_splits}
#     data_dict = load_dataset("json",data_files=data_files, num_proc=system_args.num_proc)
#     data_dict = data_dict.map(
#         lambda example: add_system_prompt(
#             example,
#             system_prompt=model_args.prompt_template,
#             use_system_prompt=model_args.use_system_prompt
#         ),
#         num_proc=system_args.num_proc
#     )
#     return data_dict


def data_prepare(
    data_splits,
    data_args: DataArgs,
    model_args: ModelArgs,
    tokenizer: AutoTokenizer,
):
    data_dict = {}
    for split in data_splits:
        data_file = os.path.join(data_args.data_dir, f"{split}.json")
        dataset = CustomDataset(
            fname=data_file,
            tokenizer=tokenizer,
            igonore_index=data_args.label_pad_token_id,
            prompt=model_args.prompt_template,
            use_system_prompt=model_args.use_system_prompt
        )
        data_dict[split] = dataset
    return data_dict


# @torch.inference_mode()
# def generate_answer(
#     model,
#     tokenizer,
#     input_ids,
#     terminators,
#     model_args: ModelArgs,
# ):

#     device = next(model.parameters()).device
#     input_ids = input_ids.to(device)

#     outputs = model.generate(
#         input_ids.unsqueeze(0),
#         max_new_tokens=model_args.max_new_tokens,
#         eos_token_id=terminators,
#         pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
#         repetition_penalty=model_args.repetition_penalty,
#         # temperature=model_args.temperature,
#         # top_p=model_args.top_p,
#         # top_k=model_args.top_k,
#         do_sample=model_args.do_sample,
#     )

#     gen_tokens = outputs[0][input_ids.size(0):]
#     text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

#     if text.startswith("[|assistant|]"):
#         text = text[len("[|assistant|]"):].lstrip()
#     if text.startswith("assistant\n\n"):
#         text = text[len("assistant\n\n"):]
#     if text.startswith("답변: "):
#         text = text[4:]
#     elif text.startswith("답변:"):
#         text = text[3:]
#     if "#" in text:
#         text = text.split("#", 1)[0].strip()
#     return text

import re
import torch

@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    input_ids,
    terminators,
    model_args,
):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    outputs = model.generate(
        input_ids.unsqueeze(0),
        max_new_tokens=model_args.max_new_tokens,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        repetition_penalty=model_args.repetition_penalty,
        do_sample=model_args.do_sample,
        # temperature=model_args.temperature,
        # top_p=model_args.top_p,
        # top_k=model_args.top_k,
    )

    gen_tokens = outputs[0][input_ids.size(0):]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # 전처리: 프리픽스 제거
    for prefix in ["[|assistant|]", "assistant\n\n", "답변:", "답변: "]:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()

    if "#" in text:
        text = text.split("#", 1)[0].strip()

    # 추론 부분 추출
    reasoning = None
    match_reasoning = re.search(r"<추론>(.*?)</추론>", text, re.DOTALL)
    if match_reasoning:
        reasoning = match_reasoning.group(1).strip()

    # ➤ CoT인 경우: <답변>...</답변> 안의 텍스트만 추출
    if model_args.is_cot:
        match_answer = re.search(r"<답변>(.*?)</답변>", text, re.DOTALL)
        if match_answer:
            text = match_answer.group(1).strip()

    return text, reasoning

def count_trainable_params(model):
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(f"Trainable: {name}, shape={param.shape}")
            total += param.numel()
    print(f"Total trainable params: {total}")

def prepare_model_tokenmizer(
    model_args: ModelArgs,
    bnb_config: BitsAndBytesConfig,
    is_train: bool = True,
    gradient_checkpointing: bool = False,
):

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id.value,
        torch_dtype=model_args.dtype.value,
        attn_implementation="flash_attention_2" if model_args.use_flash_attn2 else "eager",
        quantization_config=bnb_config,
        device_map="auto" if not model_args.use_accelerate else None,
        low_cpu_mem_usage=True if not model_args.use_accelerate else False,
        trust_remote_code=True,
    )

    # 2-stage 이상일 때 PEFT 모델 불러오기
    if model_args.current_stage != "":
        adapter_dir = os.path.join(model_args.prev_stage_model_dir, model_args.load_model)
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
        )
        for name, param in model.named_parameters():
            if "lora_" in name or "lora" in name:  # 일부 모델에서는 접두어 다름
                param.requires_grad = True
        printi(f"Loaded PEFT model from {adapter_dir}")
        model.print_trainable_parameters()

    if is_train:
        # gradient checkpointing 사용시 use_cache 사용 불가능
        if gradient_checkpointing:
            model.config.use_cache = False
        else:
            model.config.use_cache = True
        model.train()
        printi("Model is set to training mode.")
    else:
        model.config.use_cache = True
        model.eval()
        printi("Model is set to evaluation mode.")

    # model.config.torch_dtype = model_args.dtype.value

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_id.value,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    count_trainable_params(model)
    print("Model and tokenizer prepared successfully.")

    return model, tokenizer
