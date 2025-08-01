
import yaml, torch
from torch import dtype
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from peft import TaskType

GLOBAL_BATCH_SIZE = 1
NUM_DEVICES = 1
VERSION = "1-3-cot"
FIT = "V2_3.0_cot"
LORA_RANK = 64
DROPOUT = 0.1

# tensorboard --log_dir ~ --port 6006
class ModelId(Enum):
    """
    모델 ID를 정의하는 Enum 클래스입니다.
    """
    EXAONE3_5_IT_7_8B = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    #EXAONE3_5_IT_32B = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
    #QWEN2_5_LEAD_14B = "v000000/Qwen2.5-14B-Gutenberg-1e-Delta"
    KANANA1_5_IT_8B = "kakaocorp/kanana-1.5-8b-instruct-2505"
    QWEN2_5_IT_14B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

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
    additional_info: str = f"merge_no_aug_datasets_{VERSION}_early_r_{LORA_RANK}_dropout_{DROPOUT}_dosample_x_epoch_10_max_length_x_b_1_{FIT}"
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
    test: bool = False if train else True
    num_proc: int = 4
    result_save_dir_rag: str = "pre_result_with_rag"
    dpo_dataset_create_mode: bool = False
    use_rag: bool = False  # RAG 사용 여부


@dataclass
class ModelArgs:
    model_id: ModelId = ModelId.QWEN2_5_IT_14B
    dtype: DType = DType.FP16
    use_flash_attn2: bool = True
    max_new_tokens: int = 512
    do_sample: bool = False
    top_p: float = 0.8
    temperature: float = 0.7
    repetition_penalty: float = 1.05
    prompt_template: str = (
        "You are a helpful AI assistant. Please answer the user's questions kindly. " #  Think about it step by step. "
        "당신은 도움이 되는 어시스턴트입니다. 출력은 <추론>추론내용</추론><답변><답변내용></답변> 형태로 작성하세요. "
        "당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. "
        "사용자의 질문에 대해 친절하게 답변해주세요. 단, 동일한 문장을 절대 반복하지 마시오."
    )
    is_test_and_drop_other_info: bool = False


    use_system_prompt: bool = True
    early_stopping: int | bool = 3 # 5
    use_accelerate: bool = False
    load_model: str = "lora_adapter" # "lora_adapter"
    is_cot: bool = True
    current_stage: str = ""  # "1-stage", "2-stage", "3-stage", "4-stage"


@dataclass
class DataArgs:
    pad_to_multiple_of: int | None = None
    label_pad_token_id: int = -100
    # base_data_dir: str = f"datasets/merged_dataset_no_aug_v{VERSION}"
    base_data_dir: str = f"datasets/merged_dataset_no_aug_v{VERSION}"

    @property
    def data_dir(self):
        """SystemArgs의 use_rag 설정에 따라 적절한 데이터 경로 반환"""
        # SystemArgs 인스턴스 가져오기 (main 함수에서 설정됨)
        if hasattr(self, '_use_rag') and self._use_rag:
            return f"{self.base_data_dir}_for_rag"
        return self.base_data_dir

    def set_use_rag(self, use_rag: bool):
        """use_rag 설정을 저장"""
        self._use_rag = use_rag



@dataclass
class LoraArgs:
    task_type: TaskType = TaskType.CAUSAL_LM
    r: int = LORA_RANK # 128
    lora_alpha: int = LORA_RANK # 128
    lora_dropout: float = DROPOUT # 0.05
    # target_modules: list[str] | str = "all-linear"
    # target_modules: list[str] | str = field(default_factory=lambda: [
    #     'q_proj','k_proj','v_proj','o_proj' # ,'gate_proj','down_proj','up_proj', 'lm_head'
    # ])
    target_modules: str = "all-linear"
    bias: str = "none"  # or "all", "lora_only"
    use_dora: bool = False
    use_rslora: bool = False

@dataclass
class BitsAndBytesArgs:
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = DType.NF4.value
    bnb_4bit_compute_dtype: dtype = DType.FP16.value
    bnb_4bit_quant_storage: dtype = DType.FP16.value

    load_in_8bit: bool = False
    # bnb_8bit_compute_dtype: dtype = DType.FP16.value


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



# config.py에 추가할 DPO 관련 클래스들

@dataclass
class DPOTrainingArgs:
    output_dir: str = "output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    eval_accumulation_steps: int = 1
    gradient_accumulation_steps: int = GLOBAL_BATCH_SIZE // (per_device_train_batch_size * NUM_DEVICES)
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    eval_steps: int | None = 613
    save_steps: int | None = 613
    logging_steps: int = 50
    learning_rate: float = 5e-7
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    save_total_limit: int = 1
    logging_dir: str = "logs"
    report_to: list[str] | None = field(default_factory=lambda: ["tensorboard"])
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    optim: str = "adamw_torch"
    dataloader_drop_last: bool = True
    remove_unused_columns: bool = False  # DPO에서는 False로 설정

    # DPO 특화 파라미터들 (이 파라미터들은 DPOTrainer에서 별도로 처리됨)
    beta: float = 0.1
    loss_type: str = "sigmoid"
    reference_free: bool = False
    max_length: int = 2048
    max_prompt_length: int = 1024
    # max_target_length: int = 1024


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
    chunk_size: int = 512
    chunk_overlap: int = 20
    model_name: str = "nlpai-lab/KURE-v1" # "dragonkue/bge-m3-ko" # "jhgan/ko-sroberta-multitask"
    batch_size: int = 512
    index_base: str = "rag_flat.index"
    meta_base: str = "rag_meta.jsonl"
    top_k: int = 3



# @dataclass
# class FSDPArgs:
#     backward_prefetch: str = "backward_pre"
#     forward_prefetch: bool = False
#     use_orig_params: bool = False
#     sync_module_states: bool = True
#     cpu_ram_efficient_loading: bool = True
#     activation_checkpointing: bool = True
