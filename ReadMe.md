## 소개
2025년 국립국어원 AI말평 경진대회 기간 동안 LLM 파인튜닝을 위해 개발하였습니다.

### 주요 개발
1. 한국문화 질의응답(나 유형)을 위해 기존 데이터 세트와 함께 학습할 외부 데이터를 활용하였습니다.
2. **kanana-1.5-8b-instruct-2505** 모델을 기반으로 제공되는 데이터 세트와 8가지 데이터(아래 참조)와 함께 학습하였을 때, CLIcK 그리고 ETRI_QA 데이터 세트가 성능 개선 효과를 보여주었습니다.
3. 외부 데이터 세트와 함께 병합된 데이터로 **kakaocorp/kanana-1.5-8b-base** 모델에 LoRA를 적용하여 SFT를 진행하였습니다.

## 사용 방법
### 1. 학습 방법
1. src/configs/config.py 파일에서 적절히 파라미터를 수정한다. 특히, SystemArgs의 train 파라미터는 **True**로 설정한다
2. run/finetune_lora.sh 실행

### 2. 평가 방법
1. output 폴더에 학습 결과가 저장되며, 저장된 폴더내의 backup/config.py 파일 내용을 복사하여 src/configs/config.py로 붙여넣는다.
2. config.py의 SystemArgs의 train 파라미터는 **False**로 설정한다
3. run/finetune_lora.sh 실행

### 주의 사항
1. 학습과 평가를 진행할 때 파라미터의 데이터 경로를 정확하게 설정한다.

## 도커 환경에서 실행하기
- 아래 레퍼지토리 참고
```
https://github.com/wndudwkd003/kr_llm_ft_25
```

## 폴더 구조
```
.
|-- ReadMe.md
|-- count.py
|-- datasets
|-- output
|   `-- kakaocorp_kanana-1.5-8b-base_sft_lora_1-3_early_r_64_alpha_64_dropout_0.1_v1_1_fp16
|       |-- README.md
|       |-- backup
|       |   `-- config.py
|       |-- checkpoint-2452
|       |   |-- README.md
|       |   |-- adapter_config.json
|       |   |-- adapter_model.safetensors
|       |   |-- chat_template.jinja
|       |   |-- optimizer.pt
|       |   |-- rng_state.pth
|       |   |-- scaler.pt
|       |   |-- scheduler.pt
|       |   |-- special_tokens_map.json
|       |   |-- tokenizer.json
|       |   |-- tokenizer_config.json
|       |   |-- trainer_state.json
|       |   `-- training_args.bin
|       |-- logs
|       |   |-- events.out.tfevents.1752901819.c0e82bb3dc9c.3349842.0
|       |   `-- events.out.tfevents.1752905716.c0e82bb3dc9c.3402928.0
|       |-- lora_adapter
|       |   |-- README.md
|       |   |-- adapter_config.json
|       |   `-- adapter_model.safetensors
|       |-- loss_curve.png
|       `-- pred_result_last
|           `-- kakaocorp_kanana-1.5-8b-base_sft_lora_1-3_early_r_64_alpha_64_dropout_0.1_v1_1_fp16.json
|-- run
|   |-- dpo_dataset_create.sh
|   |-- finetune_2-stage.sh
|   |-- finetune_acc_lora.sh
|   |-- finetune_dpo.sh
|   |-- finetune_lora.sh
|   |-- finetune_rag.sh
|   |-- preprossing_rag.sh
|   |-- preprossing_rag_2.sh
|   |-- rag_build_db.sh
|   |-- rag_inference.sh
|   `-- rag_test.sh
|-- scripts
|   |-- click_dataset_preprocessing.py
|   |-- compare_test_json.py
|   |-- count_dataset.py
|   |-- data_compare.py
|   |-- dev_dup_remove.py
|   |-- download_datasets.py
|   |-- dpo_dataset_merge.py
|   |-- encode_pred_output.py
|   |-- etri_a_dataset_preprocessing.py
|   |-- etri_b_dataset_preprocessing.py
|   |-- etri_c_dataset_preprocessing.py
|   |-- etri_mrc_data_preprocessing.py
|   |-- for_rag_preprocessing\ copy.py
|   |-- for_rag_preprocessing.py
|   |-- for_rag_preprocessing_b.py
|   |-- gpt4o_cot_preprocessing.py
|   |-- gpt4o_cot_refiner.py
|   |-- gpt4o_cot_refiner_hard_negative.py
|   |-- gpt4o_rag_query.py
|   |-- kmmlu_dataset_preprocessing.py
|   |-- koalpaca_dataset_preprocessing.py
|   |-- korwiki_tq_dataset_preprocessing.py
|   |-- kowiki_download.py
|   |-- merge_abc_dataset.py
|   |-- merge_dataset.py
|   |-- namuwiki_download.py
|   |-- preprocessing_data.py
|   |-- refining_data.py
|   |-- search_test.py
|   |-- squad_dataset_preprocessing.py
|   |-- step_qa_dataset_create_1.py
|   |-- target_encode_pred_output.py
|   `-- tydiqa_dataset_preprocessing.py
`-- src
    |-- configs
    |   |-- accelerate_config.yaml
    |   |-- config\ copy\ 2.py
    |   |-- config\ copy\ 3.py
    |   |-- config\ copy.py
    |   |-- config.py
    |   |-- config_last.py
    |   |-- token.yaml
    |   `-- zero3_offload.json
    |-- data
    |   `-- record.py
    |-- test
    |   |-- build_db.py
    |   |-- rag_test.py
    |   |-- retriever.py
    |   `-- test_with_rag.py
    |-- train
    |   |-- train_dpo.py
    |   |-- train_sft\ copy.py
    |   |-- train_sft.py
    |   |-- train_sft_2stage-rag.py
    |   |-- train_sft_dpo_1_dataset_create.py
    |   |-- train_sft_dpo_dataset_create.py
    |   `-- train_sft_last.py
    `-- utils
        |-- checker.py
        |-- dpo_dataset.py
        |-- dpo_dataset_utils.py
        |-- log_utils.py
        |-- model_utils.py
        |-- path_utils.py
        |-- print_utils.py
        |-- qa_dataset\ copy\ 2.py
        |-- qa_dataset\ copy\ 3.py
        |-- qa_dataset\ copy\ 4_rag.py
        |-- qa_dataset\ copy.py
        |-- qa_dataset.py
        |-- qa_dataset_backup.py
        |-- qa_dataset_heareheare_73.py
        |-- qa_dataset_last.py
        |-- qa_dataset_temp\ copy.py
        `-- two_stage_rag_dataset.py
```

## 시도해본 것 들
1. 허깅페이스에 올라온 공식 QWEN2.5 14B모델과 한국어 리더보드에 올라온 QWEN2.5 14B 기반 모델을 QLoRA로 학습해보았으나 간혹 중국어 답변과 할루시네이션을 보여주어 좋은 결과를 보여주지 못했습니다.
2. GPT 4.1 API를 사용하여 질문-답변 쌍을 주고 이러한 답변이 나오는 논리적 근거를 추론하라는 지시를 통해 CoT 데이터를 구성하여 카나나 모델에 학습해보았으나 좋은 결과는 보여주지 못했습니다. (추론시에는 올바른 연결고리를 생성하였으나 정작 답변시에는 이상한 답변을 보여주었음. 또한, 엉뚱하게 추론하는 경우도 존재하였음)
3. **LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct** 모델로도 LoRA 파인튜닝을 진행해보았으나 좋은 결과를 보여주지 못했습니다.
4. RTX3090 여러대를 활용하여 높은 용량의 모델을 학습하려고 하였으나 각종 버그와 개발 실력 이슈(ㅠㅠ) 때문에 진행할 수 없었습니다.
5. 데이터 증강을 해보았으나 오히려 좋지 않은결과가 나타났습니다.


## 결과
- [2025]한국문화 질의응답(나 유형) 예선에서 최종 점수 73.3754271로 7위에서 탈락하였습니다.
