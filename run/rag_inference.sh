#!/bin/bash

# CUDA_VISIBLE_DEVICES=0: 사용할 GPU 지정

CUDA_VISIBLE_DEVICES=1 python -m src.test.test_with_rag

