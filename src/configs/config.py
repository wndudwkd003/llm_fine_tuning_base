from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import torch
from torch import dtype
import yaml
from peft import TaskType, LoraConfig
from transformers import TrainingArguments, BitsAndBytesConfig

GLOBAL_BATCH_SIZE = 8
NUM_DEVICES = 1


class ModelId(Enum):
    """
    모델 ID를 정의하는 Enum 클래스입니다.
    """
    EXAONE3_5_IT_7_8B = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    EXAONE3_5_IT_2_4B = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"


class DType(Enum):
    """
    데이터 타입을 정의하는 Enum 클래스입니다.
    """
    FP16 = torch.float16
    BF16 = torch.bfloat16
    FP32 = torch.float32
    TF32 = torch.float32
    NF4 = "nf4"


@dataclass
class SystemArgs:
    additional_info: str = "trial-1"
    gpu_number: int = 0
    seed: int = 42
    hf_token: str = yaml.safe_load(open("src/configs/token.yaml", "r"))["hf_token"]
    backup_path: List[str] = field(default_factory=lambda: [
        "src/configs/config.py",
    ])
    use_lora: bool = True
    use_qlora: bool = True

    train: bool = False
    test: bool = True
    num_proc: int = 4


@dataclass
class ModelArgs:
    model_id: ModelId = ModelId.EXAONE3_5_IT_7_8B
    dtype: DType = DType.BF16
    use_flash_attn2: bool = True
    max_seq_length: int = 4096
    max_new_tokens: int = 512
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.8
    temperature: float = 0.7
    repetition_penalty = 1.05
    prompt_template: str = (
        "You are a helpful AI assistant. Please answer the user's questions kindly. "
        "당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. "
        "사용자의 질문에 대해 친절하게 답변해주세요. 단, 동일한 문장을 절대 반복하지 마시오."
    )


@dataclass
class DataArgs:
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    num_workers: int = 4
    data_dir: str = "datasets/refine_sub_3_data_korean_culture_qa_V1.0"




@dataclass
class LoraArgs:
    task_type: TaskType = TaskType.CAUSAL_LM
    r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        'q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head'
    ])
    bias: str = "none"  # or "all", "lora_only"

@dataclass
class BitsAndBytesArgs:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = DType.NF4.value
    bnb_4bit_compute_dtype: dtype = DType.BF16.value
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingArgs:
    output_dir: str = "output"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = GLOBAL_BATCH_SIZE // (per_device_train_batch_size * NUM_DEVICES)
    eval_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 50
    learning_rate: float = 1e-4
    num_train_epochs: int = 10
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    save_total_limit: int = 1
    logging_dir: str = "logs"
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    fp16: bool = False
    bf16: bool = True
