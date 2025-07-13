#!/bin/bash

# CUDA_VISIBLE_DEVICES=0: 사용할 GPU 지정

CUDA_VISIBLE_DEVICES=2 python -m src.train.train_sft_dpo_1_dataset_create

