#!/bin/bash

uv run src/generation_evaluate.py \
    --input_file data/exp-hallu/output/model_generation/google--gemma-2-2b-it__with_passage__1_sampled_triviaqa-train_20k_positive_negative_generation.jsonl \
    --batch_eval \
    --create_plots \
    --use_wandb