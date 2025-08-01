

def check_sft_type(use_lora: bool, use_qlora: bool, user_dora: bool, use_rslora: bool) -> str:
    if user_dora:
        return "sft_dora"

    elif use_rslora:
        return "sft_rslora"

    elif use_lora:
        if use_qlora:
            return "sft_qlora"
        else:
            return "sft_lora"
    else:
        return "sft"
