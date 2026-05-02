#! /bin/bash

# Example script to run the model activation extractor

uv run src/get_model_activation.py \
    --model_name gemma-2-2b-it \
    --data_file output/model_generation/google--gemma-2-2b-it__with_passage_check_relevance__1_dev_data_2k_rel_irrel_1_generation_filtered.jsonl \
    --output_dir output/model_activations \
    --prompt_type with_passage \
    --pos_to_analyze="-1" \
    --use_wandb
