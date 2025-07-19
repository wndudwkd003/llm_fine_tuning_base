from dataclasses import dataclass, field

@dataclass
class Metadata:
    corpus: str
    split: str
    title: str
    text: str  # 검색용 완전 전처리 텍스트
    # model_text: str  # LLM 모델용 가벼운 전처리 텍스트
    chunk_id: int = -1

# @dataclass
# class Record:
#     text: str = field(default="")
#     metadata: Metadata = field(default_factory=Metadata)
