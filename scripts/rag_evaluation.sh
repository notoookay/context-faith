#!/bin/bash

uv run src/rag_evaluation.py \
    --model_name gemma-2-2b-it \
    --input_file DATA_FILE \
    --batch_size 32 \
    --create_plots \
    --use_wandb \
