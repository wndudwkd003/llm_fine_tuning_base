

def check_sft_type(use_lora: bool, use_qlora: bool) -> str:
    if use_lora:
        if use_qlora:
            return "sft_qlora"
        else:
            return "sft_lora"
    else:
        return "sft"
