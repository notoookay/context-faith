#! /bin/bash

uv run eval/faitheval_direction.py \
    --model_name gemma-2-2b-it \
    --task_type unanswerable \
    --direction_file output/directions/gemma-2-2b-it_with_passage_irrelevant__gemma-2-2b-it_with_passage_relevant/diff_in_mean_-1.pt \
    --coefficient="2.0" \
    --pos_to_apply="-1" \
    --batch_size 1 \
    --layer 15 \
    --use_wandb
