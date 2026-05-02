#!/bin/bash


uv run src/model_generate.py \
    --model_name gemma-2-2b-it \
    --input_file output/model_generation/google--gemma-2-2b-it__with_passage_check_relevance__1_train_data_10k_rel_irrel_2_generation_filtered.jsonl \
    --batch_size 16 \
    --prompt_type with_passage \
    --use_wandb \
    --fp16