import os, shutil

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from src.configs.config import Config, ModelId, DType
from src.utils.path_utils import create_out_dir


def add_system_prompt(example):
    msgs = example["messages"]
    if msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": config.prompt_template}] + msgs
    return {"messages": msgs}



def main(config: Config):

    # 1) 저장 경로 및 백업 설정
    create_out_dir(config.output_dir, config.backup_path)

    # 2) 데이터 세트 로드
    data_files = {sp: os.path.join(config.data_dir, f"{sp}.json") for sp in ["train", "dev"]}
    data_dict = load_dataset("json",data_files=data_files)


    # 3) 모델, 토크나이저
    model = AutoModelForCausalLM.from_pretrained(
        model_id=config.model_id.value,
        torch_dtype=config.dtype.value,
        attn_implementation="flash_attention_2" if config.flash_attn2 else "eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )





if __name__ == "__main__":
    config = Config()
    os.environ["HF_TOKEN"] = config.hf_token
    main(config)


