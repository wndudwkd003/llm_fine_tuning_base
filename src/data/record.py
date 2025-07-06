from dataclasses import dataclass, field

@dataclass
class Metadata:
    corpus: str = field(default="")
    split: str = field(default="")
    title: str = field(default="")

# @dataclass
# class Record:
#     text: str = field(default="")
#     metadata: Metadata = field(default_factory=Metadata)
