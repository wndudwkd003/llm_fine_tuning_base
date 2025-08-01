#!/bin/bash

# CUDA_VISIBLE_DEVICES=0: 사용할 GPU 지정

CUDA_VISIBLE_DEVICES=0 python -m src.train.train_sft
# sleep 14400 && CUDA_VISIBLE_DEVICES=1 python -m src.train.train_sft
