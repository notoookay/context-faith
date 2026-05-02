#!/bin/bash

uv run src/knockout_heads.py \
    --model_name google/gemma-2-9b-it \
    --data_file data/exp-hallu/output/model_generation/google--gemma-2-9b-it__with_passage_check_relevance__1_dev_data_2k_rel_irrel_1_generation_filtered.jsonl \
    --batch_size 4 \
    --top_n_heads 20 \
    --output_file knockout_results.json