#!/bin/bash

# CUDA_VISIBLE_DEVICES=0: 사용할 GPU 지정

CUDA_VISIBLE_DEVICES=1 python -m scripts.for_rag_preprocessing
