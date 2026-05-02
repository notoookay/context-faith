#! /bin/bash

uv run src/binary_classifier.py \
    --load_activations_group1_train data/exp-hallu/output/model_activations/gemma-2-9b-it_google--gemma-2-9b-it__with_passage_check_relevance__1_train_data_10k_rel_irrel_2_generation_filtered_with_passage_layers_all_pos_-1_activation/activations.pt \
    --load_activations_group2_train data/exp-hallu/output/model_activations/gemma-2-9b-it_google--gemma-2-9b-it__with_passage_check_relevance__1_train_data_10k_rel_irrel_1_generation_filtered_with_passage_layers_all_pos_-1_activation/activations.pt \
    --load_activations_group1_test data/exp-hallu/output/model_activations/gemma-2-9b-it_google--gemma-2-9b-it__with_passage_check_relevance__1_dev_data_2k_rel_irrel_2_generation_filtered_with_passage_layers_all_pos_-1_activation/activations.pt \
    --load_activations_group2_test data/exp-hallu/output/model_activations/gemma-2-9b-it_google--gemma-2-9b-it__with_passage_check_relevance__1_dev_data_2k_rel_irrel_1_generation_filtered_with_passage_layers_all_pos_-1_activation/activations.pt \
    --group1_name gemma-2-9b-it-irrelevant \
    --group2_name gemma-2-9b-it-relevant \
    --save_classifiers \
    --create_heatmap \
    --use_wandb
