#!/bin/bash

uv run src/get_generation_entropy.py \
    --input_file output/direction_generation/gemma-2-2b-it/google--gemma-2-2b-it__with_passage_check_relevance__1_dev_data_2k_rel_irrel_2_generation_filtered_with_passage_add_coef0.0_pos-1_layer15.jsonl \
    --model_name gemma-2-2b-it \
    --layer 15 \
    --coefficient "0.0" \
    --use_wandb
    