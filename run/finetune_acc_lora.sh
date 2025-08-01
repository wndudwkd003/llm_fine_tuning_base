#!/bin/bash

# CUDA_VISIBLE_DEVICES=0: 사용할 GPU 지정

# CUDA_VISIBLE_DEVICES=0,3 python -m src.train.train_sft

CUDA_VISIBLE_DEVICES=0,1,3 accelerate launch -m src.train.train_sft
# --config_file /workspace/llm_fine_tuning_base/src/configs/accelerate_config.yaml
