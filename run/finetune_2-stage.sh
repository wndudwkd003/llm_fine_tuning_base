#!/bin/bash

# CUDA_VISIBLE_DEVICES=0: 사용할 GPU 지정

CUDA_VISIBLE_DEVICES=0 python -m src.train.train_sft_2stage-rag
#sleep 3600 && CUDA_VISIBLE_DEVICES=0 python -m src.train.train_sft
