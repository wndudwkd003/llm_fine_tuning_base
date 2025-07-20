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


# def data_prepare(
#     data_splits,
#     data_args: DataArgs,
#     model_args: ModelArgs,
#     tokenizer: AutoTokenizer,
# ):
#     data_dict = {}
#     for split in data_splits:
#         data_file = os.path.join(data_args.data_dir, f"{split}.json")
#         dataset = CustomDataset(
#             fname=data_file,
#             tokenizer=tokenizer,
#             igonore_index=data_args.label_pad_token_id,
#             prompt=model_args.prompt_template,
#             use_system_prompt=model_args.use_system_prompt
#         )
#         data_dict[split] = dataset
#     return data_dict



# def data_prepare(
#     splits,
#     data_args,
#     model_args,
#     tokenizer,
#     retriever=None,  # RAG retriever 추가
#     use_rag=False,   # RAG 사용 여부
#     rag_top_k=5      # 검색할 문서 수
# ):
#     from src.utils.qa_dataset import CustomDataset

#     data_dict = {}
#     for split in splits:
#         fname = os.path.join(data_args.data_dir, f"{split}.json")
#         data_dict[split] = CustomDataset(
#             fname,
#             tokenizer,
#             retriever=retriever,  # retriever 전달
#             use_rag=use_rag,      # RAG 사용 여부
#             top_k=rag_top_k,      # 검색할 문서 수
#             igonore_index=data_args.label_pad_token_id,
#             prompt=model_args.prompt_template,
#             use_system_prompt=model_args.use_system_prompt
#         )
#     return data_dict



def data_prepare(
    splits,
    data_args,
    model_args,
    tokenizer,
    use_rag=False,   # RAG 사용 여부
    is_test_and_drop_other_info=False,
):
    from src.utils.qa_dataset import CustomDataset

    data_dict = {}
    for split in splits:
        fname = os.path.join(data_args.data_dir, f"{split}.json")
        data_dict[split] = CustomDataset(
            fname,
            tokenizer,
            use_rag=use_rag,      # RAG 사용 여부
            igonore_index=data_args.label_pad_token_id,
            prompt=model_args.prompt_template,
            use_system_prompt=model_args.use_system_prompt,
            is_test_and_drop_other_info=is_test_and_drop_other_info
        )
        # stats = data_dict[split].check_token_lengths()
        # print(f"Dataset {split} stats: {stats}")
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

from typing import List, Union, Optional

def _sanitize_eos(
    terminators: Optional[Union[int, List[Optional[int]]]],
    tokenizer,
) -> Union[int, List[int]]:
    if terminators is None:
        return tokenizer.eos_token_id

    if isinstance(terminators, (list, tuple)):
        cleaned = [tid for tid in terminators if tid is not None]
        if not cleaned:
            return tokenizer.eos_token_id
        return cleaned if len(cleaned) > 1 else cleaned[0]

    return terminators

@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    input_ids,
    # terminators,
    model_args,
):

    device = next(model.parameters()).device

    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_ids = input_ids.to(device)

    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    pad_id = tokenizer.pad_token_id

    outputs = model.generate(
        input_ids.unsqueeze(0),
        max_new_tokens=model_args.max_new_tokens,
        attention_mask=attention_mask.unsqueeze(0),
        pad_token_id=pad_id,
        repetition_penalty=model_args.repetition_penalty,
        do_sample=model_args.do_sample,
        # temperature=model_args.temperature,
        # top_p=model_args.top_p,
        # top_k=model_args.top_k,
    )

    # 전체 출력에서 assistant 이후 부분 추출
    gen_tokens = outputs[0, input_ids.size(0):]
    full_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    # print(f"Full output: {full_output}")  # 디버깅용 출력

    if "[|assistant|]" in full_output:
        text = full_output.split("[|assistant|]", 1)[1].strip()
    elif "assistant\n" in full_output:
        text = full_output.split("assistant\n", 1)[1].strip()
    elif "assistant" in full_output:
        text = full_output.split("assistant", 1)[1].strip()
    # else:
    #     # fallback
    #     gen_tokens = outputs[0][input_ids.size(0):]
    #     text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    else:
        text = full_output.strip()

    # 나머지 처리 로직...
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
        # model_max_length=8096
    )


    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = model.config.max_position_embeddings

    model.config.pad_token_id = tokenizer.pad_token_id

    # count_trainable_params(model)
    # print("Model and tokenizer prepared successfully.")
    # print(f"config.max_position_embeddings: {model.config.max_position_embeddings}")
    # print(f"config.max_sequence_length: {getattr(model.config, 'max_sequence_length', 'Not found')}")
    # print(f"config.n_positions: {getattr(model.config, 'n_positions', 'Not found')}")
    # print(f"model_max_length: {tokenizer.model_max_length}")
    # print(f"max_position_embeddings: {getattr(tokenizer, 'max_position_embeddings', 'Not found')}")
    return model, tokenizer
