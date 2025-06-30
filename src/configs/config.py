from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import torch
import yaml

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


@dataclass
class Config:
    """
    (설정) 주석이 붙은 설정을 변경하여 사용하세요.
    """
    # 허깅페이스 설정 (설정)
    hf_token: str = yaml.safe_load(open("src/configs/token.yaml", "r"))["hf_token"]

    # 모델 설정 (설정)
    model_id: ModelId = ModelId.EXAONE3_5_IT_7_8B
    dtype: DType = DType.BF16
    flash_attn2: bool = True

    # 학습 설정 (설정)
    num_train_epochs: int = 10
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True

    # 로라 설정 (설정)
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # 로깅 설정 (설정)
    logging_steps: int = 1

    # 배치 사이즈 설정 (설정)
    global_batch_size: int = 64
    batch_per_device: int = 2
    num_devices: int = 1

    # DeepSpeed 타입 설정 (설정)
    # src/configs 폴더에 있는 JSON 파일을 지정합니다.
    deepspeed: str|None = None # "src/configs/zero3_offload.json"

    # 데이터셋 설정 (설정)
    # dev.json
    data_dir: str = "datasets/refine_sub_3_data_korean_culture_qa_V1.0"


    # 결과 저장 설정 (설정)
    save_description: str = "exaone-3.5-7.8B-it-tri-1"
    output_dir: str = f"output/sft_lora/{save_description}"

    # 모델 프롬프트 설정 (설정)
    prompt_template: str = (
        "You are a helpful AI assistant. Please answer the user's questions kindly. "
        "당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. "
        "사용자의 질문에 대해 친절하게 답변해주세요. 단, 동일한 문장을 절대 반복하지 마시오."
    )

    # 그래디언트 누적 설정
    grad_accum_steps: int = global_batch_size // (batch_per_device * num_devices)


    # 기타 설정
    backup_path: List[str] = field(default_factory=lambda: [
        "src/configs/config.py",
    ])





