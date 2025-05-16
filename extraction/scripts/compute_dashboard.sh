#!/bin/bash

python3 extraction/compute_dashboard.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --sae_path Gubrg/deepSeek-R1-Distill-Qwen-1.5B-openthoughts \
    --dataset_path Gubrg/OpenThoughts-10k-DeepSeek-R1 \
    --scores_dir extraction/scores \
    --sae_id blocks.19.hook_resid_post \
    --topk 20 \
    --n_samples 10000 \
    --minibatch_size_features 128 \
    --minibatch_size_tokens 64 \
    --output_dir extraction/dashboards \
