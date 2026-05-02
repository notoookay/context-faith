#! /bin/bash

uv run src/extract_diff_in_mean.py \
    --load_activations_group1 output/model_activations/gemma-2-2b-it_google--gemma-2-2b-it__with_passage_check_relevance__1_train_data_10k_rel_irrel_2_generation_filtered_with_passage_layers_all_pos_-1_activation/activations.pt \
    --load_activations_group2 output/model_activations/gemma-2-2b-it_google--gemma-2-2b-it__with_passage_check_relevance__1_train_data_10k_rel_irrel_1_generation_filtered_with_passage_layers_all_pos_-1_activation/activations.pt \
    --group1_name gemma-2-2b-it_with_passage_irrelevant \
    --group2_name gemma-2-2b-it_with_passage_relevant \
    --visualize