#! /bin/bash

uv run src/context_utility_evaluate.py \
    --input_file output/direction_generation/gemma-2-2b-it/google--gemma-2-2b-it__with_passage_check_relevance__1_dev_data_2k_rel_irrel_1_generation_filtered_with_passage_add_coef2.0_pos-1_layer15.jsonl \
    --llm_eval_model google/gemma-2-2b-it \
    --create_plots \
    --use_wandb
