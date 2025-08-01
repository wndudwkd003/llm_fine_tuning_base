import os, json, re
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
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
from src.utils.print_utils import printi, printe
from src.utils.model_utils import (
    initialize_config,
    data_prepare,
    generate_answer,
    prepare_model_tokenmizer
)
from src.test.retriever import Retriever

LABEL_PAD_TOKEN_ID = -100




def extract_question(user_msg):
    question_blocks = re.findall(r"\[질문\](.*?)(?=\[|$)", user_msg, re.DOTALL)
    if not question_blocks:
        return None
    return question_blocks[-1].strip()

def prepare_prompt_with_rag(messages, question, retrieved):
    background = "\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(retrieved)]) if retrieved else ""
    last_user_msg = messages[-1].copy()
    if background:
        last_user_msg["content"] = f"[배경]\n{background.strip()}\n\n{last_user_msg['content'].strip()}"
    else:
        last_user_msg["content"] = last_user_msg["content"].strip()
    return messages[:-1] + [last_user_msg]

def generate_prompt_ids(tokenizer, messages):
    return tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

def run_rag_inference(
    model,
    tokenizer,
    retriever,
    sample,
    terminators,
    model_args,
    top_k=5
):
    user_msg = sample["messages"][-1]["content"]
    question = extract_question(user_msg)
    if question is None:
        printe(f"[질문] 항목을 찾을 수 없음: {sample.get('id', 'UNKNOWN')}")
        return None

    retrieved = retriever.retrieve(question, top_k=top_k)

    messages = prepare_prompt_with_rag(
        sample["messages"],
        question,
        retrieved
    )

    prompt_ids = generate_prompt_ids(
        tokenizer,
        messages
    )

    answer = generate_answer(
        model,
        tokenizer,
        prompt_ids[0],
        terminators,
        model_args
    )

    return {"answer": answer}

def save_results(results, save_dir, target_name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{target_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    printi(f"Inference finished. Predictions saved to: {path}")


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

    model, tokenizer = prepare_model_tokenmizer(
        model_args,
        bnb_config,
        is_train=False,  # Inference mode
        is_gradient_checkpointing=sft_training_args.gradient_checkpointing
    )

    adapter_dir = os.path.join(sft_training_config.output_dir, model_args.load_model)
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
    )

    retriever = Retriever(rag_index_args)

    data_dict = data_prepare(
        ["test"],
        data_args,
        system_args,
        sft_training_args,
        model_args
    )

    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
    if eot_id: terminators.append(eot_id)

    results = []
    for sample in tqdm(data_dict["test"], desc="Inference with RAG"):
        
        output = run_rag_inference(
            model,
            tokenizer,
            retriever,
            sample,
            terminators,
            model_args,
            top_k=rag_index_args.top_k
        )

        if output:
            sample_copy = sample.copy()
            sample_copy["output"] = output
            results.append(sample_copy)

    save_results(
        results,
        os.path.join(sft_training_config.output_dir, system_args.result_save_dir_rag),
        target_name
    )



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
