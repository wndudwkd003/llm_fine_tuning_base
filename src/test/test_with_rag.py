import os, random, json

from tqdm.auto import tqdm

import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)

from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as LCFAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document

from src.configs.config import (
    SystemArgs,
    ModelArgs,
    DataArgs,
    SFTTrainingArgs,
    BitsAndBytesArgs,
    LoraArgs,
    RAGIndexArgs
)

from src.utils.seeds import set_seed
from src.utils.path_utils import create_out_dir
from src.utils.checker import check_sft_type
from src.utils.log_utils import save_training_curves
from src.utils.print_utils import printw, printi, printe
from src.utils.model_utils import (
    initialize_config,
    data_prepare
)

LABEL_PAD_TOKEN_ID = -100

@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt_ids,
    terminators,
    model_args: ModelArgs,
    device="cuda",
):

    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(prompt_ids)

    outputs = model.generate(
        input_ids=prompt_ids.unsqueeze(0),
        attention_mask=attention_mask.to(device).unsqueeze(0),
        do_sample=model_args.do_sample,
        max_new_tokens=model_args.max_new_tokens,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        temperature=model_args.temperature,
        top_p=model_args.top_p,
        top_k=model_args.top_k,
        repetition_penalty=model_args.repetition_penalty,
    )

    gen_tokens = outputs[0][prompt_ids.size(0):]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    if text.startswith("[|assistant|]"):
        text = text[len("[|assistant|]"):].lstrip()
    if text.startswith("assistant\n\n"):
        text = text[len("assistant\n\n"):]
    if text.startswith("답변: "):
        text = text[4:]
    elif text.startswith("답변:"):
        text = text[3:]
    if "#" in text:
        text = text.split("#", 1)[0].strip()
    return text




def main(
    system_args: SystemArgs,
    model_args: ModelArgs,
    data_args: DataArgs,
    sft_training_args: SFTTrainingArgs,
    bits_and_bytes_args: BitsAndBytesArgs,
    lora_args: LoraArgs,
    rag_index_args: RAGIndexArgs
):
    global LABEL_PAD_TOKEN_ID
    LABEL_PAD_TOKEN_ID = data_args.label_pad_token_id

    printi("Strating Inference with RAG")

    output_dir, target_name = create_out_dir(
        sft_training_args.output_dir,
        check_sft_type(system_args.use_lora, system_args.use_qlora),
        model_args.model_id.value,
        system_args.additional_info,
        backup_path=system_args.backup_path
    )

    sft_training_args.output_dir = output_dir
    sft_training_args.logging_dir = os.path.join(output_dir, "logs")

    bnb_config, lora_config, sft_training_config = initialize_config(
        system_args,
        bits_and_bytes_args,
        lora_args,
        sft_training_args
    )

    data_dict = data_prepare(["test"], data_args)



if __name__ == "__main__":
    system_args = SystemArgs()
    model_args = ModelArgs()
    data_args = DataArgs()
    sft_training_args = SFTTrainingArgs()
    bits_and_bytes_args = BitsAndBytesArgs()
    lora_args = LoraArgs()
    rag_index_args = RAGIndexArgs()

    set_seed(system_args.seed)
    os.environ["HF_TOKEN"] = system_args.hf_token

    main(
        system_args,
        model_args,
        data_args,
        sft_training_args,
        bits_and_bytes_args,
        lora_args,
        rag_index_args
    )
