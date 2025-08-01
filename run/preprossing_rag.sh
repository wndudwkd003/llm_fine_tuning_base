#!/bin/bash

# CUDA_VISIBLE_DEVICES=0: 사용할 GPU 지정

# CUDA_VISIBLE_DEVICES=1 python -m scripts.for_rag_preprocessing
# sleep 3600 && CUDA_VISIBLE_DEVICES=0 python -m scripts.for_rag_preprocessing --original_dir "datasets/merged_dataset_no_aug_v1-3_remove_duplication"
#sleep 14400 && CUDA_VISIBLE_DEVICES=0 python -m scripts.for_rag_preprocessing --original_dir "datasets/merged_dataset_no_aug_v1-3_remove_duplication"

CUDA_VISIBLE_DEVICES=0 python -m scripts.for_rag_preprocessing --original_dir "datasets/merged_dataset_no_aug_v1-3_remove_duplication"
