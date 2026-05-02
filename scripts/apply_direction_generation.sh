#! /bin/bash

uv run src/apply_direction_generation.py \
    --input_file output/model_generation/google--gemma-2-2b-it__with_passage_check_relevance__1_dev_data_2k_rel_irrel_1_generation_filtered.jsonl \
    --model_name gemma-2-2b-it \
    --prompt_type with_passage \
    --direction_file output/directions/gemma-2-2b-it_with_passage_irrelevant__gemma-2-2b-it_with_passage_relevant/diff_in_mean_-1.pt \
    --pos_to_apply="-1" \
    --layer "15" \
    --coefficient="2.0" \
    --use_wandb
