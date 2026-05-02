#! /bin/bash

uv run src/attn_head_knockout.py \
    --input_file output/model_generation/google--gemma-2-2b-it__with_passage_check_relevance__1_dev_data_2k_rel_irrel_1_generation_filtered.jsonl \
    --model_name gemma-2-2b-it \
    --prompt_type with_passage \
    --batch_size 4 \
    --max_new_tokens 100 \
    --use_wandb