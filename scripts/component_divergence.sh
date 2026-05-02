#! /bin/bash

uv run src/component_divergence.py \
    --load_activations_group1 data/exp-hallu/output/model_activations/gemma-2-9b-it_google--gemma-2-9b-it__with_passage_check_relevance__1_train_data_10k_rel_irrel_1_generation_filtered_with_passage_layers_all_pos_-1_activation/activations.pt \
    --load_activations_group2 data/exp-hallu/output/model_activations/gemma-2-9b-it_google--gemma-2-9b-it__with_passage_check_relevance__1_train_data_10k_rel_irrel_2_generation_filtered_with_passage_layers_all_pos_-1_activation/activations.pt \
    --group1_name gemma-2-9b-it-relevant \
    --group2_name gemma-2-9b-it-irrelevant \
    --top_k 20 \
    --position="-1" \
    --create_heatmap \
    --plot_distribution