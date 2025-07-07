import os

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
