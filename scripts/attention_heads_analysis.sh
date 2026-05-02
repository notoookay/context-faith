#!/bin/bash

uv run src/attention_heads_analysis.py \
    --data_file output/model_generation/google--gemma-2-2b-it__with_passage_check_relevance__1_train_data_10k_rel_irrel_1_generation_filtered.jsonl \
    --model_name gemma-2-2b-it \
    --save_heatmap \
    --top_k 5 \
    # --direction_file output/directions/gemma-2-2b-it_with_passage_irrelevant__gemma-2-2b-it_with_passage_relevant/diff_in_mean_-1.pt \
    # --coefficient 2.0 \
    # --layer 15
