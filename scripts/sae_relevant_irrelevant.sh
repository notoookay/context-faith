#! /bin/bash

uv run src/sae_relevant_irrelevant.py \
    --model_name llama-3.1-8b-instruct \
    --sae_repo_id llama_scope_lxr_8x \
    --relevant_data_file RELEVANT_DATA_FILE \
    --irrelevant_data_file IRRELEVANT_DATA_FILE \
    --load_relevant_activations LOAD_RELEVANT_ACTIVATIONS_FILE \
    --load_irrelevant_activations LOAD_IRRELEVANT_ACTIVATIONS_FILE \
    --output_dir ./output/sae_relevant_irrelevant \
    --prompt_type with_passage \
    --pos_to_analyze="-1" \
    --use_wandb
