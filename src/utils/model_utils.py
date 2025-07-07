import os
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig
from datasets import load_dataset

from src.configs.config import (
    SystemArgs,
    SFTTrainingArgs,
    BitsAndBytesArgs,
    LoraArgs,
    DataArgs,
    ModelArgs
)

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


def add_system_prompt(example, system_prompt: str, use_system_prompt: bool = True):
    if use_system_prompt:
        example['messages'] = [{"role": "system", "content": system_prompt}] + example['messages']
    else:
        example['messages'][0]['content'] = system_prompt + "\n" + example['messages'][0]['content']
    return example



def data_prepare(
    data_splits,
    data_args: DataArgs,
    system_args: SystemArgs,
    model_args: ModelArgs
):
    data_files = {sp: os.path.join(data_args.data_dir, f"{sp}.json") for sp in data_splits}
    data_dict = load_dataset("json",data_files=data_files, num_proc=system_args.num_proc)
    data_dict = data_dict.map(
        lambda example: add_system_prompt(
            example,
            system_prompt=model_args.prompt_template,
            use_system_prompt=model_args.use_system_prompt
        ),
        num_proc=system_args.num_proc
    )
    return data_dict




@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt_ids,
    terminators,
    model_args: ModelArgs,
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

