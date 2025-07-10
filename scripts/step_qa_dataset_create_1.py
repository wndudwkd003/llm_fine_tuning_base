import os, json, shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
MODEL_CURTURE = "SEOKDONG/llama3.1_korean_v1.4_sft_by_aidx"

def create_curture_context(data, tokenizer, model, device, tgt_path):
    model.to(device)
    model.eval()

    processed = []

    for item in data:
        da_input = item["input"]
        da_output = item["output"]

        user_prompt = (
            f"[데이터 세트 기타 정보] 카테고리: {da_input.get('category', '')}, 도메인: {da_input.get('domain', '')}, 키워드: {da_input.get('topic_keyword', '')}\n"
            f"[데이터 세트 질문] {da_input.get('question', '')}\n"
            f"[데이터 세트 답변] {da_output.get('answer', '')}\n"
            f"[주의 사항] 생성된 배경 지식은 데이터 세트 증강을 위해 사용됩니다. 부정확한 정보는 배제하고, 정확하고 유용하며, 자세한 배경 지식을 생성하세요. 특수기호 사용을 최소화하고 자연스러운 한국어로 작성하세요.\n"
            f"[주의 사항] 생성된 배경 지식은 [데이터 세트 질문]에 대한 답변을 제공하는 것이 아니라, 질문에 대한 배경 지식을 제공해야합니다.\n"
            f"[주의 사항] 동일한 답변을 반복하지 마세요. 배경 지식은 다양하고 풍부해야 합니다. 다시 한번 강조합니다. [실제 질의]에 대한 답변을 출력하세요. [데이터 세트 질문]과 [데이터 세트 답변]에 대한 배경 지식을 생성해야합니다.\n"
            f"[주의 사항] 출력은 '[배경 지식]'으로 시작하는 한 문단만 생성하세요. 다른 정보는 포함하지 마세요.\n"
            f"[실제 질의] '데이터 세트 질문'에 대한 답변은 '데이터 세트 답변'입니다. 이러한 답변이 나오기 위해 '데이터 세트 기타 정보'를 바탕으로 필요한 [배경 지식]을 생성하세요.\n"
        )


        inputs = tokenizer(user_prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("[RAW 생성된 텍스트]:", generated_text)

        if generated_text.startswith(user_prompt):
            generated_text = generated_text[len(user_prompt):].strip()

        background_and_extra = extract_background_and_extra(generated_text)

        print(f"[생성된 배경 지식]: {background_and_extra[0]}")
        print(f"[추가 설명]: {background_and_extra[1]}")
        item["curture_answer"] = background_and_extra

        processed.append(item)

        # 파일에 바로 저장
        with open(tgt_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)


def extract_background_and_extra(text):
    background_match = re.search(r'\[배경 지식\](.*?)(\[|$)', text, re.DOTALL)
    extra_match = re.search(r'\[추가 설명\](.*?)(\[/추가 설명\]|\[|$)', text, re.DOTALL)

    background = background_match.group(1).strip() if background_match else ""
    extra = extra_match.group(1).strip() if extra_match else ""

    return [background, extra]



if __name__ == "__main__":
    source_data_dir = "datasets/sub_3_data_korean_culture_qa_V1.0"
    target_data_dir = source_data_dir + "_step_qa"
    target_files = ["train.json", "dev.json", "test.json"]

    os.makedirs(target_data_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CURTURE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_CURTURE, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for file_name in target_files:
        src_path = os.path.join(source_data_dir, file_name)
        tgt_path = os.path.join(target_data_dir, file_name)

        if file_name == "test.json":
            shutil.copyfile(src_path, tgt_path)
        else:
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            create_curture_context(data, tokenizer, model, device, tgt_path)
