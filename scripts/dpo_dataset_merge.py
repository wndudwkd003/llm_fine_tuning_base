import os, json

choose_data_dir = "datasets/etri_qa_abc_cot_refined_4.1"
stage_3_data_dir = "datasets/etri_qa_abc_cot_3-stage_refined_4.1"

target = ["train.json"]
output_data_dir = choose_data_dir + "_3-stage_for_dpo"
os.makedirs(output_data_dir, exist_ok=True)

for t in target:
    print(f"Processing {t}...")

    with open(os.path.join(choose_data_dir, t), "r", encoding="utf-8") as f:
        choose_data = json.load(f)



    with open(os.path.join(stage_3_data_dir, t), "r", encoding="utf-8") as f:
        stage_3_data = json.load(f)

    for i in range(len(choose_data)):
        choose_data[i]["output"]["reject"] = []
        choose_data[i]["output"]["reject"].extend(
            stage_3_data[i]["output"]["cot_answer"]
        )

    save_path = os.path.join(output_data_dir, t)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(choose_data, f, ensure_ascii=False, indent=4)

    print(f"Saved {t} to {save_path}")
