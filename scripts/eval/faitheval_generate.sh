#!/bin/bash

# Run the evaluation
uv run eval/faitheval_generate.py \
    --model_name gemma-2-2b-it \
    --task_type unanswerable \
    --batch_size 2 \
    --use_wandb