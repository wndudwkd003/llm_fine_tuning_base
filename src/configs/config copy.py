
import yaml, torch
from torch import dtype
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from peft import TaskType

GLOBAL_BATCH_SIZE = 1
NUM_DEVICES = 1
VERSION = 1

# tensorboard --log_dir ~ --port 6006
class ModelId(Enum):
    """
    모델 ID를 정의하는 Enum 클래스입니다.
    """
    EXAONE3_5_IT_7_8B = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    #EXAONE3_5_IT_32B = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
    #QWEN2_5_LEAD_14B = "v000000/Qwen2.5-14B-Gutenberg-1e-Delta"
    KANANA1_5_IT_8B = "kakaocorp/kanana-1.5-8b-instruct-2505"

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
    additional_info: str = f"merge_no_aug_datasets_{VERSION}_early_r_64_dosample_o_epoch_10_max_length_x_b_1"
    seed: int = 42
    hf_token: str = yaml.safe_load(open("src/configs/token.yaml", "r"))["hf_token"]
    backup_path: list[str] = field(default_factory=lambda: [
        "src/configs/config.py",
    ])
    use_lora: bool = True
    use_qlora: bool = False
    # 반드시 train 또는 test는 하나만 true로 설정할 것
    # True or False
    train: bool = False
    test: bool = True
    num_proc: int = 4
    result_save_dir_rag: str = "pre_result_with_rag"
    dpo_dataset_create_mode: bool = False


@dataclass
class ModelArgs:
    model_id: ModelId = ModelId.KANANA1_5_IT_8B
    dtype: DType = DType.FP16
    use_flash_attn2: bool = True
    max_new_tokens: int = 2048
    do_sample: bool = False
    top_p: float = 0.8
    temperature: float = 0.7
    repetition_penalty: float = 1.05
    prompt_template: str = (
        "You are a helpful AI assistant. Please answer the user's questions kindly. "#  Think about it step by step. "
        "당신은 도움이 되는 어시스턴트입니다. "
        "당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. "
        "사용자의 질문에 대해 친절하게 답변해주세요. 단, 동일한 문장을 절대 반복하지 마시오."
    )
    use_system_prompt: bool = True
    early_stopping: int | bool = 3 # 5
    use_accelerate: bool = False
    load_model: str = "lora_adapter" # "lora_adapter"
    current_stage: str = ""
    is_cot: bool = False  # True or False, CoT 사용 여부


@dataclass
class DataArgs:
    pad_to_multiple_of: int | None = None
    label_pad_token_id: int = -100
    # data_dir: str = "datasets/refine_sub_3_data_korean_culture_qa_V1.0"
    data_dir: str = f"datasets/merged_dataset_no_aug_v{VERSION}"


@dataclass
class LoraArgs:
    task_type: TaskType = TaskType.CAUSAL_LM
    r: int = 64 # 128
    lora_alpha: int = 64 # 128
    lora_dropout: float = 0.0 # 0.05
    # target_modules: list[str] | str = "all-linear"
    target_modules: list[str] | str = field(default_factory=lambda: [
        'q_proj','k_proj','v_proj','o_proj' # ,'gate_proj','down_proj','up_proj', 'lm_head'
    ])

    #  "all-linear"
    bias: str = "none"  # or "all", "lora_only"
    use_dora: bool = False  # True or False, DORA 사용 여부

@dataclass
class BitsAndBytesArgs:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = DType.NF4.value
    bnb_4bit_compute_dtype: dtype = DType.FP16.value
    bnb_4bit_use_double_quant: bool = True
    # bnb_4bit_quant_storage: dtype = DType.FP16.value


@dataclass
class SFTTrainingArgs:
    output_dir: str = "output"
    num_train_epochs: int = 10                # Epochs to train the model
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    eval_accumulation_steps: int = 1
    gradient_accumulation_steps: int = GLOBAL_BATCH_SIZE // (per_device_train_batch_size * NUM_DEVICES)
    eval_strategy: str = "steps" # "no", "epoch", "steps"
    save_strategy: str = "steps" # "no", "epoch", "steps"
    eval_steps: int | None = 613 # 100
    save_steps: int | None = 613 # 100
    logging_steps: int = 50
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    save_total_limit: int = 1
    logging_dir: str = "logs"
    report_to: list[str] | None = field(default_factory=lambda: ["tensorboard"])
    fp16: bool = True
    bf16: bool = False
    packing: bool = False
    # max_length: int = 4096
    gradient_checkpointing: bool = True
    activation_offloading: bool = False
    label_names: list[str] = field(default_factory=lambda: ["labels"])
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    optim: str = "adamw_torch" # "adamw_torch" is default, adamw_hf or "adamw_8bit" or "paged_adamw_8bit"


@dataclass
class RAGIndexArgs:
    raw_text_dir: list[dict] = field(default_factory=lambda:[
        {
            "dir": "datasets/namuwikitext",
            "base": "20200302",
            "ext": ["train", "dev", "test"]
        },
        {
            "dir": "datasets/kowikitext",
            "base": "20200920",
            "ext": ["train", "dev", "test"]
        },
    ])
    # version: str   = "20200302"    # 파일명 날짜
    index_dir: str = "rag_index"
    chunk_size: int = 258
    chunk_overlap: int = 32
    model_name: str = "nlpai-lab/KURE-v1" # "dragonkue/bge-m3-ko" # "jhgan/ko-sroberta-multitask"
    batch_size: int = 256
    index_base: str = "rag_flat.index"
    meta_base: str = "rag_meta.jsonl"
    top_k: int = 5



# @dataclass
# class FSDPArgs:
#     backward_prefetch: str = "backward_pre"
#     forward_prefetch: bool = False
#     use_orig_params: bool = False
#     sync_module_states: bool = True
#     cpu_ram_efficient_loading: bool = True
#     activation_checkpointing: bool = True
