#!/bin/bash

uv run src/get_top5_prob_sum.py \
    --input_file output/direction_generation/gemma-2-2b-it/google--gemma-2-2b-it__with_passage_check_relevance__1_dev_data_2k_rel_irrel_1_generation_filtered_with_passage_add_coef2.0_pos-1_layer15.jsonl \
    --model_name gemma-2-2b-it \
    --direction_file output/directions/gemma-2-2b-it_with_passage_irrelevant__gemma-2-2b-it_with_passage_relevant/diff_in_mean_-1.pt \
    --layer 15 \
    --coefficient "2.0"